#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chuanliantui 串联腿轮足机器人 sim2sim 部署验证脚本
  由 mj_sim2sim.py(imcawl 版)复刻,契约常量按 chuanliantui 配置改写（Isaac Gym -> MuJoCo）

用途
    在 MuJoCo 里加载训练好的策略(chuanliantui 任务)，复现训练时完全一致的
    观测构造、历史缓冲、动作->力矩映射与控制时序，验证策略在 MuJoCo 串联代理
    中的行为。该路径只用于训练代理一致性回放，不代表真实闭链机构或真机。

串联代理契约（对齐 wheel_legged_gym/envs/base/legged_robot.py）
    - 策略：ActorCriticSequence。action = actor(cat(obs_25, encoder(history_125)))
      导出的 policy_1.pt 只含 actor、缺 encoder，不可用；因此直接加载 model_*.pt 完整权重。
    - DOF 顺序：[rf0, rf1, rfwheel, lf0, lf1, lfwheel]
    - 观测 25 维：[base_ang_vel*0.25(3), projected_gravity(3), cmd*[2.0,0.25,5.0](3),
                   腿关节 pos [rf0,rf1,lf0,lf1](4), dof_vel*0.05(6), last_action(6)]，裁剪 ±100
    - 历史 125=25*5：FIFO，最旧在前、最新在末尾；上电用首帧重复 5 次填充
    - 串联代理 dof_vel 用位置差分；闭链默认复现 H7 五连杆解算和反馈速度路径。
      --closed_chain_controller cad 可回到原 CAD 解算与混合差分速度路径。
    - 动作->串联训练代理力矩：腿位置控制(Kp=10,Kd=1)，轮速度控制(Kp=0,Kd=0.1)；
      六维力矩直接写入同名训练 DOF 的 MuJoCo motor，不做闭链 Jacobian 映射。
    - 时序：sim_dt=0.002，decimation=5 -> 策略 100Hz，PD 内环 500Hz
    - 力矩上限：[40,40,3.9,40,40,3.9] N·m
    - 串联代理与闭链都独立施加每侧 150 N 气弹簧伸张力；
      --gas_spring_force 0 可复现旧串联代理的无弹簧条件，不占用策略电机限矩。
    - 动作延迟：训练随机化 0–10 ms；本脚本默认零延迟，属于无延迟对照。
    - 前进方向:机体 +x(与训练 tracking_lin_vel 的 base_lin_vel[:,0] 一致)。
      cmd_vx 是策略命令通道 0，对应机体系 vx。
    - 训练 `heading_command=False`，命令通道 1 是偏航角速度，直接写入 cmd_yaw；
      不使用航向保持外环。
    - H7 闭链控制另外采用腿角差环绕、轮目标 ±20 rad/s 和十拍零动作预热。
      CAD 后轴角采用名义几何注册，不等同于实机编码器标定；详见 H7 适配器。

运行
    /home/zn/miniforge3/envs/wheellegged_py38/bin/python sim2sim/mj_sim2sim_ct.py \
        --checkpoint logs/chuanliantui/Sep08_12-56-46_new1_train_proxy_v1_resume/model_3000.pt --render
    先测起立：--standup --cmd_vx 0 --cmd_height 0.22
    再测行走：--cmd_vx 1.0
    需要先在该环境安装 mujoco：pip install mujoco
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 允许从仓库根目录导入 wheel_legged_gym
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence
from sim2sim.chuanliantui_closed_adapter import ClosedChainAdapter


# --------------------------------------------------------------------------------------
# 契约常量（全部已从 config / legged_robot.py 核实）
# --------------------------------------------------------------------------------------
JOINT_NAMES = ["rf0", "rf1", "rfwheel", "lf0", "lf1", "lfwheel"]
ACTUATOR_NAMES = ["{}_motor".format(name) for name in JOINT_NAMES]
NUM_ACTIONS = 6
NUM_OBS = 25
# 平地特权观测 = base_lin_vel(3)+actor obs(25)+last_actions(12)+dof_acc(6)+height(1)+
# torques(6)+mass(1)+com(3)+default_dof_offset(6)+friction(1)+restitution(1) = 65。
NUM_PRIVILEGED_OBS = 65
OBS_HISTORY_LENGTH = 5
NUM_ENCODER_OBS = NUM_OBS * OBS_HISTORY_LENGTH  # 125
LATENT_DIM = 3

DEFAULT_DOF_POS = np.array([-0.06, 0.10, 0.0, 0.06, -0.10, 0.0], dtype=np.float64)
LEG_POSITION_INDICES = np.array((0, 1, 3, 4), dtype=np.intp)
# 当前起立训练从 0.15 m 的地面后摆初态开始。配置写入的连续关节角会在
# MuJoCo 写入前规范到等价的 [-pi, pi) 表示；首次轮接地前策略不推理。
STANDUP_START_HEIGHT = 0.15
STANDUP_INITIAL_DOF_POS = np.array(
    [-1.566, 0.0, 0.0, 1.566, 0.0, 0.0], dtype=np.float64
)
WHEEL_CONTACT_FORCE_THRESHOLD = 1.0
P_GAINS = np.array([10.0, 10.0, 0.0, 10.0, 10.0, 0.0], dtype=np.float64)
D_GAINS = np.array([1.0, 1.0, 0.1, 1.0, 1.0, 0.1], dtype=np.float64)
TORQUE_LIMITS = np.array([40.0, 40.0, 3.9, 40.0, 40.0, 3.9], dtype=np.float64)

POS_ACTION_SCALE = 0.5
VEL_ACTION_SCALE = 10.0

OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
# commands_scale = [lin_vel, ang_vel, height_measurements]
COMMANDS_SCALE = np.array([2.0, 0.25, 5.0], dtype=np.float64)

CLIP_OBS = 100.0
CLIP_ACTIONS = 100.0

SIM_DT = 0.002
DECIMATION = 5  # 策略 100Hz，PD 内环 500Hz

# 策略网络结构（来自 legged_robot_config.py:class policy）
ACTOR_HIDDEN_DIMS = [128, 64, 32]
CRITIC_HIDDEN_DIMS = [256, 128, 64]
ENCODER_HIDDEN_DIMS = [128, 64]

DEFAULT_MODEL_XML = os.path.join(_THIS_DIR, "chuanliantui_train_proxy.xml")
DEFAULT_CLOSED_MODEL_XML = os.path.join(_THIS_DIR, "chuanliantui.xml")
GAS_SPRING_ACTUATOR_NAMES = (
    "right_gas_spring_motor",
    "left_gas_spring_motor",
)
DEFAULT_GAS_SPRING_FORCE = 150.0


# --------------------------------------------------------------------------------------
# 数学工具（自实现，避免依赖 isaacgym，使脚本可独立运行）
# --------------------------------------------------------------------------------------
def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """把世界系向量 v 旋转到机体系（等价于 isaacgym.quat_rotate_inverse）。

    q_xyzw: [x, y, z, w]（与 Isaac Gym root_states[3:7] 一致）
    v:      [3]
    公式与 isaacgym.torch_utils.quat_rotate_inverse 完全一致。
    """
    q_vec = q_xyzw[:3]
    q_w = q_xyzw[3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    return a - b + c


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """把角度差绕到 [-pi, pi)，与 legged_robot.compute_dof_vel 的 remainder 逻辑一致。"""
    return np.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo 四元数为 [w,x,y,z]，转成 Isaac Gym 用的 [x,y,z,w]。"""
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


# --------------------------------------------------------------------------------------
# 策略加载
# --------------------------------------------------------------------------------------
def load_policy(checkpoint_path: str, device: str = "cpu") -> ActorCriticSequence:
    """用仓库的 ActorCriticSequence 类重建网络并载入 model_*.pt 权重。"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"{checkpoint_path} 不含 model_state_dict；请使用 logs/.../model_*.pt 完整 checkpoint，"
            f"而非导出的 policy_1.pt。"
        )
    state_dict = ckpt["model_state_dict"]
    expected_shapes = {
        "encoder.0.weight": (ENCODER_HIDDEN_DIMS[0], NUM_ENCODER_OBS),
        "actor.0.weight": (ACTOR_HIDDEN_DIMS[0], NUM_OBS + LATENT_DIM),
        "critic.0.weight": (CRITIC_HIDDEN_DIMS[0], NUM_PRIVILEGED_OBS + LATENT_DIM),
    }
    for name, expected_shape in expected_shapes.items():
        if name not in state_dict or tuple(state_dict[name].shape) != expected_shape:
            actual_shape = None if name not in state_dict else tuple(state_dict[name].shape)
            raise ValueError(
                "checkpoint 观测接口不匹配：{} 为 {}，当前 chuanliantui 需要 {}。"
                "轮位置已从 actor 观测删除（25 维、历史 125），请使用重新训练的 25 维 model_*.pt。".format(
                    name, actual_shape, expected_shape
                )
            )
    ac = ActorCriticSequence(
        num_obs=NUM_OBS,
        num_critic_obs=NUM_PRIVILEGED_OBS + LATENT_DIM,  # 65+3=68（仅 critic 用，推理不涉及）
        num_actions=NUM_ACTIONS,
        num_encoder_obs=NUM_ENCODER_OBS,
        latent_dim=LATENT_DIM,
        encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
        actor_hidden_dims=ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=CRITIC_HIDDEN_DIMS,
        activation="elu",
    )
    ac.load_state_dict(state_dict)
    ac.to(device)
    ac.eval()
    return ac


# --------------------------------------------------------------------------------------
# 观测构造与力矩计算（严格对齐 legged_robot.py）
# --------------------------------------------------------------------------------------
def build_obs(base_quat_xyzw, base_ang_vel_body, dof_pos, dof_vel, commands, last_action):
    """构造 25 维本体观测，连续轮的位置被明确删除。

    注意：base_ang_vel_body 已是机体系角速度（由 mj_objectVelocity local 帧取得），
    无需再做 quat_rotate_inverse。projected_gravity 才需要把世界重力转到机体系。
    """
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    projected_gravity = quat_rotate_inverse(base_quat_xyzw, gravity_world)

    obs = np.concatenate([
        base_ang_vel_body * OBS_SCALE_ANG_VEL,            # 3
        projected_gravity,                                # 3 (无缩放)
        commands[:3] * COMMANDS_SCALE,                    # 3
        (dof_pos[LEG_POSITION_INDICES] - DEFAULT_DOF_POS[LEG_POSITION_INDICES]) * OBS_SCALE_DOF_POS,  # 4
        dof_vel * OBS_SCALE_DOF_VEL,                      # 6
        last_action,                                      # 6 (原始未缩放)
    ]).astype(np.float64)
    obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
    assert obs.shape == (NUM_OBS,), obs.shape
    return obs


def compute_torques(action, dof_pos, dof_vel, wheel_vel_limit=None):
    """动作->关节力矩；默认对齐 legged_robot._compute_torques。

    腿关节(0,1,3,4)：位置控制，目标角 = action*0.5 + default。
    轮关节(2,5)：速度控制，目标速度 = action*10.0，Kp=0。
    wheel_vel_limit 仅用于复现实机速度目标限幅的诊断对照。
    """
    action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)

    pos_ref = action * POS_ACTION_SCALE
    pos_ref[2] = 0.0
    pos_ref[5] = 0.0

    vel_ref = action * VEL_ACTION_SCALE
    vel_ref[0] = 0.0
    vel_ref[1] = 0.0
    vel_ref[3] = 0.0
    vel_ref[4] = 0.0
    if wheel_vel_limit is not None:
        vel_ref[[2, 5]] = np.clip(vel_ref[[2, 5]], -wheel_vel_limit, wheel_vel_limit)

    torques = P_GAINS * (pos_ref + DEFAULT_DOF_POS - dof_pos) + D_GAINS * (vel_ref - dof_vel)
    return np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)


class H7PolicyHistory:
    """H7 有效观测 FIFO 和十个策略步的零动作预热。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = np.zeros(NUM_ENCODER_OBS, dtype=np.float64)
        self.ready = False
        self.warmup_count = 0

    def update(self, obs, valid):
        if not valid or not np.isfinite(obs).all():
            self.reset()
            return self.history.copy(), False
        if not self.ready:
            self.history = np.tile(obs, OBS_HISTORY_LENGTH)
            self.ready = True
        else:
            self.history = np.concatenate((self.history[NUM_OBS:], obs))
        active = self.warmup_count >= 10
        self.warmup_count = min(10, self.warmup_count + 1)
        return self.history.copy(), active


# --------------------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------------------
def run(args):
    import glfw
    import mujoco

    model_xml = args.model_xml or (
        DEFAULT_CLOSED_MODEL_XML if args.closed_chain else DEFAULT_MODEL_XML
    )
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")
    if not os.path.isfile(model_xml):
        raise FileNotFoundError(f"MJCF 不存在: {model_xml}")

    device = "cpu"
    policy = load_policy(args.checkpoint, device=device)
    print(f"已加载策略: {args.checkpoint}")

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    if args.closed_chain and model.neq != 4:
        raise ValueError(
            "--closed_chain 需要含 4 个 connect 的真实闭链 XML；"
            f"当前 neq={model.neq}: {model_xml}"
        )
    if not args.closed_chain and model.neq != 0:
        raise ValueError(
            "该脚本已关闭真实闭链路径，只接受 neq=0 的串联训练代理 MJCF；"
            "如需旧闭链验证，请显式传入 --closed_chain。"
        )
    model.opt.timestep = SIM_DT
    if args.friction is not None:
        # MuJoCo 接触摩擦取两 geom 的逐元素最大值；统一改滑动摩擦即可控制轮地摩擦。
        # 训练等效摩擦区间约 [0.4, 1.0]（机器人 [0.3,1.5] 与地面 0.5 取 PhysX 平均），均值 0.75。
        model.geom_friction[:, 0] = args.friction
        print(f"已覆盖所有 geom 滑动摩擦 = {args.friction}")
    print(
        f"已加载{'真实闭链验证模型' if args.closed_chain else '串联训练代理'}: {model_xml} "
        f"(nq={model.nq}, nv={model.nv}, nu={model.nu}, neq={model.neq})"
    )

    qpos_adr = np.zeros(NUM_ACTIONS, dtype=np.int32)
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"MJCF 中找不到关节: {name}")
        qpos_adr[i] = model.jnt_qposadr[jid]

    mujoco.mj_forward(model, data)
    adapter = None
    h7_adapter = None
    h7_history = None
    h7_output_ready = False
    if args.closed_chain:
        if args.closed_chain_controller == "h7":
            from sim2sim.chuanliantui_h7_adapter import H7ClosedChainAdapter
            h7_adapter = H7ClosedChainAdapter(mujoco, model, data, rear_zero=args.h7_rear_zero,
                                              wheel_vel_limit=args.wheel_vel_limit)
            adapter = h7_adapter
            h7_history = H7PolicyHistory()
        else:
            adapter = ClosedChainAdapter(mujoco, model, data)
        print("chuanliantui 闭链已启用：rf0/rf00 与 lf0/lf00 是实体电机。")
        print("闭链控制路径：{}".format(args.closed_chain_controller))
        if h7_adapter is not None:
            print("H7：五连杆 0.21/0.25 m、反馈速度、角差环绕、轮目标限幅={} rad/s、预热 10 拍。".format(args.wheel_vel_limit))
            print("后轴角使用{}；这是仿真角度注册，未经实机共同姿态标定。".format(
                "显式 --h7_rear_zero" if args.h7_rear_zero is not None else "CAD 输入杆名义方向"))
            if not np.isclose(args.cmd_height, 0.20):
                print("高度命令覆盖为 {:.3f} m；所对照 H7 源码默认为 0.20 m。".format(args.cmd_height))
    else:
        actuator_ids = np.zeros(NUM_ACTIONS, dtype=np.int32)
        for i, name in enumerate(ACTUATOR_NAMES):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise RuntimeError(f"MJCF 中找不到训练代理电机: {name}")
            actuator_ids[i] = actuator_id

    gas_spring_actuator_ids = np.empty(len(GAS_SPRING_ACTUATOR_NAMES), dtype=np.int32)
    for i, name in enumerate(GAS_SPRING_ACTUATOR_NAMES):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError("模型缺少气弹簧执行器（请同步生成代理 XML）: " + name)
        gas_spring_actuator_ids[i] = actuator_id
    print("气弹簧每侧 {:.1f} N，独立于六个策略电机施力。".format(args.gas_spring_force))

    # 基座 free joint 地址（qpos 前 7 位 = pos(3)+quat wxyz(4)，qvel 前 6 位 = linvel(3)+angvel(3)）
    base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    base_qadr = model.jnt_qposadr[base_jid]
    base_vadr = model.jnt_dofadr[base_jid]
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # PhysX 会将连续关节的状态回绕到 [-pi, pi)。例如配置中的 11 rad
    # 与 -1.566 rad 是同一姿态，但策略观测不是同一个数。MuJoCo 的 hinge
    # 不会自动回绕；若直接写 11，actor 会持续读到训练中不存在的大角度状态。
    # 在写入 MuJoCo 前统一规范化，随后观测、历史和 PD 都直接读取该状态。
    requested_initial_dof_pos = (
        STANDUP_INITIAL_DOF_POS if args.standup else args.initial_dof_pos
    )
    initial_dof_pos = wrap_to_pi(np.asarray(requested_initial_dof_pos, dtype=np.float64))
    if not np.allclose(initial_dof_pos, requested_initial_dof_pos):
        print(
            "初始关节已按 PhysX 表示规范到 [-pi, pi)："
            f" {np.array2string(initial_dof_pos, precision=6)}"
        )

    def place_collision_meshes_on_ground():
        """沿世界 z 平移 floating base，使当前碰撞网格的最低顶点刚好离地。"""
        mujoco.mj_forward(model, data)
        min_z = np.inf
        for gid in range(model.ngeom):
            if gid == floor_gid or model.geom_contype[gid] == 0:
                continue
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                raise RuntimeError("ground_start 目前只支持 mesh 碰撞 geom")
            mesh_id = model.geom_dataid[gid]
            start = model.mesh_vertadr[mesh_id]
            count = model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start:start + count]
            rotation = data.geom_xmat[gid].reshape(3, 3)
            z_values = vertices @ rotation[2, :] + data.geom_xpos[gid, 2]
            min_z = min(min_z, float(z_values.min()))
        if not np.isfinite(min_z):
            raise RuntimeError("ground_start 未找到机器人碰撞网格")
        data.qpos[base_qadr + 2] += args.ground_clearance - min_z
        mujoco.mj_forward(model, data)
        print(f"ground_start：碰撞网格最低点 z={min_z:.6f} m，"
              f"已平移到 clearance={args.ground_clearance:.4f} m；"
              f"base_z={data.qpos[base_qadr + 2]:.6f} m")

    def reset_sim_state(standup):
        mujoco.mj_resetData(model, data)
        if h7_history is not None:
            h7_history.reset()
        if adapter is not None:
            adapter.set_virtual_pose(initial_dof_pos)
        else:
            for i, qpos in enumerate(initial_dof_pos):
                data.qpos[qpos_adr[i]] = qpos
        data.qpos[base_qadr + 2] = STANDUP_START_HEIGHT if standup else args.init_height
        data.qpos[base_qadr + 3:base_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        # mj_resetData 会清 ctrl；首帧诊断和起立预备阶段也应包含物理弹簧。
        data.ctrl[gas_spring_actuator_ids] = args.gas_spring_force
        mujoco.mj_forward(model, data)
        if args.ground_start:
            if standup:
                raise ValueError("--ground_start 不能与 --standup 同用")
            place_collision_meshes_on_ground()

    reset_sim_state(args.standup)

    # 命令 [vx, yaw_rate, height]；vx/yaw 经 cmd_delay 后在 cmd_ramp 秒内线性爬升到目标
    commands_target = np.array([args.cmd_vx, args.cmd_yaw, args.cmd_height], dtype=np.float64)

    def commands_at(t: float) -> np.ndarray:
        factor = np.clip((t - args.cmd_delay) / max(args.cmd_ramp, 1e-6), 0.0, 1.0)
        return np.array([commands_target[0] * factor,
                         commands_target[1] * factor,
                         commands_target[2]], dtype=np.float64)

    commands = commands_at(0.0)

    # ---- 键盘遥操作（渲染时默认启用）----
    # viewer 线程回调里改、主循环里读；纯 float 赋值受 GIL 保护，无需加锁。
    kb = {"vx": args.cmd_vx, "yaw_rate": args.cmd_yaw, "height": args.cmd_height, "reset_standup": None}

    def key_callback(keycode):
        # 注意：viewer 内置大量单字母快捷键（W线框/S阴影/R反射/数字键组显隐...）且无法拦截，
        # 故遥操作使用方向键/PgUp/PgDn/空格；复位须同时按 Ctrl+R。
        if keycode == 265:            # ↑ 加速
            # 上限 1.5 = 当前策略安全包线（encoder 高速估计偏置，1.8 会瞬态发散翻车，
            # 详见 2026-07-21 排查：Isaac cmd 1.8 真实速度也只到 ~1.5）
            kb["vx"] = min(kb["vx"] + 0.1, 1.5)
        elif keycode == 264:          # ↓ 减速（可到倒车）
            kb["vx"] = max(kb["vx"] - 0.1, -1.5)
        elif keycode == 263:          # ← 左转（偏航角速度 +）
            kb["yaw_rate"] = float(np.clip(kb["yaw_rate"] + 0.2, -2.0, 2.0))
        elif keycode == 262:          # → 右转
            kb["yaw_rate"] = float(np.clip(kb["yaw_rate"] - 0.2, -2.0, 2.0))
        elif keycode == 266:          # PgUp 升高
            kb["height"] = min(kb["height"] + 0.02, 0.40)
        elif keycode == 267:          # PgDn 降低
            kb["height"] = max(kb["height"] - 0.02, 0.20)
        elif keycode in (32, 257):    # 空格 / 回车 急停
            kb["vx"] = 0.0
        elif keycode == glfw.KEY_R and (
            (window := glfw.get_current_context()) is not None
            and (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                 or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)
        ):                            # Ctrl+R 恢复启动姿态（--standup 时为地面后摆）
            kb["reset_standup"] = args.standup
            target = "0.15 m 地面后摆" if args.standup else "启动站姿"
            print(f"[遥操作] Ctrl+R：恢复{target}请求...")
            return
        else:
            return
        print(f"[遥操作] vx={kb['vx']:+.1f} m/s  偏航角速度={kb['yaw_rate']:+.1f} rad/s  "
              f"高度={kb['height']:.2f} m")

    def read_dof_state():
        if h7_adapter is not None:
            return h7_adapter.read_policy_state()
        dof_pos = np.empty(NUM_ACTIONS, dtype=np.float64)
        for i in ((0, 2, 3, 5) if adapter is not None else range(NUM_ACTIONS)):
            dof_pos[i] = data.qpos[qpos_adr[i]]
        if adapter is None:
            return dof_pos, None
        # 闭链观测只用前/后实体电机状态反算虚拟膝；不读被动 rf1/lf1 的 qpos/qvel。
        knees = adapter.read_virtual_leg_state()
        dof_pos[1], dof_pos[4] = knees["right"][0], knees["left"][0]
        return dof_pos, np.array((knees["right"][1], knees["left"][1]))

    def control_outputs(action):
        if h7_adapter is not None:
            if not h7_output_ready:
                return np.zeros(NUM_ACTIONS), np.zeros(NUM_ACTIONS)
            return h7_adapter.compute_control(action)
        virtual = compute_torques(action, dof_pos, dof_vel, args.wheel_vel_limit)
        motor = adapter.map_virtual_torques(virtual) if adapter is not None else virtual
        return virtual, motor

    # 状态缓冲
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    last_dof_pos, feedback_vel = read_dof_state()
    dof_vel = feedback_vel if h7_adapter is not None else np.zeros(NUM_ACTIONS, dtype=np.float64)

    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    push_schedule = None
    push_impulse = np.zeros(3)
    if args.push_schedule:
        from sim2sim.standing_push_schedule import load
        push_schedule = load(args.push_schedule)
        if not math.isclose(push_schedule.to_dict()["physics_dt_s"], SIM_DT, abs_tol=1e-8):
            raise ValueError("推力计划的物理步长与MuJoCo不一致")
    wheel_bids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rfwheel"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lfwheel"),
    }

    def read_base_state():
        quat_wxyz = data.qpos[base_qadr + 3:base_qadr + 7].copy()
        base_quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
        # 注意：必须用 mjOBJ_XBODY（机体系 xmat/xpos）！mjOBJ_BODY 返回的是惯性主轴系
        # （ximat/xipos）的速度——base 惯量特征值降序排布导致主轴系相对机体系是轴置换，
        # 用 BODY 会把陀螺仪三分量整个换位喂给策略（2026-07-21 排查：站立勉强靠
        # projected_gravity 撑住、一加速就翻车的根因）。
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, base_bid, vel6, 1)
        base_ang_vel_body = vel6[:3].copy()
        base_lin_vel_body = vel6[3:].copy()
        return base_quat_xyzw, base_ang_vel_body, base_lin_vel_body

    def has_wheel_contact():
        """复现训练端首次轮接地判据：任一轮对地竖直接触力 > 1 N。"""
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            wheel_on_floor = (
                (contact.geom1 == floor_gid and body2 in wheel_bids)
                or (contact.geom2 == floor_gid and body1 in wheel_bids)
            )
            if wheel_on_floor:
                mujoco.mj_contactForce(model, data, contact_id, contact_force)
                if contact_force[0] > WHEEL_CONTACT_FORCE_THRESHOLD:
                    return True
        return False

    # obs_history：上电用首帧重复填充（对齐 reset_idx）
    dof_pos, _ = read_dof_state()
    base_quat_xyzw, base_ang_vel_body, base_lin_vel_body = read_base_state()
    first_obs = build_obs(base_quat_xyzw, base_ang_vel_body, dof_pos, dof_vel, commands, last_action)
    obs_history = np.tile(first_obs, OBS_HISTORY_LENGTH).astype(np.float64)  # 125
    has_landed = not args.standup

    viewer = None
    if args.render:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
            if args.gas_spring_view:
                with viewer.lock():
                    # 只改变渲染：加粗气弹簧，突出端点，并透视显示机构。
                    for side, color in (("right", [0.0, 0.9, 1.0, 1.0]),
                                        ("left", [1.0, 0.35, 0.05, 1.0])):
                        tid = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_TENDON, side + "_gas_spring_tendon")
                        if tid < 0:
                            raise ValueError("气弹簧视图缺少 tendon: " + side)
                        model.tendon_width[tid] = 0.006
                        model.tendon_rgba[tid] = color
                        for end in ("upper", "lower"):
                            sid = mujoco.mj_name2id(
                                model, mujoco.mjtObj.mjOBJ_SITE,
                                side + "_gas_spring_" + end)
                            if sid < 0:
                                raise ValueError("气弹簧视图缺少 site: " + side + "_" + end)
                            model.site_rgba[sid] = color
                            model.site_size[sid, 0] = 0.009
                            model.site_group[sid] = 5
                    viewer.opt.geomgroup[3] = 0  # 隐藏重复的碰撞网格。
                    viewer.opt.sitegroup[:] = 0
                    viewer.opt.sitegroup[5] = 1
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = base_bid
                    viewer.cam.distance = 1.1
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -15
                print("气弹簧视图：左侧青色，右侧橙色；圆点为安装端点，粗线为气弹簧轴线。")
        except Exception as e:  # noqa: BLE001
            if viewer is not None:
                viewer.close()
            print(f"警告：无法启动 viewer（{e}）；改为无渲染运行。")
            viewer = None
    if viewer is not None:
        print("遥操作模式：点击 MuJoCo 窗口获得焦点后按键控制 —— "
              "↑ 加速 0.1 | ↓ 减速 | ← 左转 | → 右转 | PgUp/PgDn 升降高度 | 空格/回车 急停 | "
              "Ctrl+R 恢复启动姿态（--standup 时为 0.15 m 地面后摆）。（单独的字母键是 viewer 内置渲染快捷键，勿用）关闭窗口退出。")

    run_mode = "渲染遥操作（关闭窗口退出）" if viewer is not None else "无渲染运行"
    print(f"开始仿真：{run_mode} "
          f"(vx={args.cmd_vx}, yaw={args.cmd_yaw}, height={args.cmd_height}, "
          f"wheel_vel_limit={args.wheel_vel_limit}, sim_time={args.sim_time})")
    print("日志字段说明：")
    print("  [步号]   策略步序号（100Hz，100 步 = 1 秒）")
    print("  x        基座世界系 x 坐标 [m]（前进方向，位置在涨=向前移动）")
    print("  z        基座离地高度 [m]（应贴住 cmd_height；跌到 ~0.12 = 翻倒扣在基座盒上）")
    print("  v_fwd    机体系前向速度 [m/s]（mj_objectVelocity 读取，应跟住 cmd_vx）")
    print("  vx_w     世界系 x 向速度 [m/s]（qvel 直读；与 v_fwd 应接近，差得远=速度读取有问题）")
    print("  轮速l/r  左/右轮等效线速度 [m/s]（轮角速度*半径0.0625；无打滑时应≈车速，")
    print("           远大于车速=轮子打滑空转，远小于=机器人在蹭移而轮子没滚）")
    print("  pg_xyz   重力在机体系的投影（直立=[0,0,-1]；本机 +x 前向，pg_x 反映前后倾，")
    print("           持续增大 = 正在倾倒；pg_z 变正 = 已完全翻转）")
    print("  |a|max   6 维动作绝对值最大者（正常 <1，持续增长/饱和 = 策略在挣扎）")

    policy_dt = SIM_DT * DECIMATION  # 每个策略步对应的仿真时间（0.01s）
    wall_start = time.perf_counter()

    step = 0
    wheel_limit_steps = 0
    wheel_limit_targets = 0
    diagnostic_rows = []
    previous_base_pos = data.qpos[base_qadr:base_qadr + 3].copy()
    joint_ids = np.array([mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES])
    dof_adr = model.jnt_dofadr[joint_ids]
    try:
        while (viewer is None or viewer.is_running()) and (
            args.sim_time is None or step * policy_dt < args.sim_time
        ):
            # ---- 遥操作复位：回目标姿态并清空控制器/历史缓冲 ----
            if viewer is not None and kb["reset_standup"] is not None:
                reset_standup = kb["reset_standup"]
                kb["reset_standup"] = None
                reset_sim_state(reset_standup)
                last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
                last_dof_pos, feedback_vel = read_dof_state()
                dof_pos = last_dof_pos.copy()
                dof_vel = feedback_vel if h7_adapter is not None else np.zeros(NUM_ACTIONS, dtype=np.float64)
                kb["vx"] = 0.0
                kb["yaw_rate"] = 0.0
                has_landed = not reset_standup
                base_quat_xyzw, base_ang_vel_body, base_lin_vel_body = read_base_state()
                obs_history = np.tile(
                    build_obs(base_quat_xyzw, base_ang_vel_body, dof_pos, dof_vel,
                              np.array([0.0, 0.0, kb["height"]]), last_action),
                    OBS_HISTORY_LENGTH).astype(np.float64)
                pose_name = "0.15 m 地面后摆" if reset_standup else "初始站姿"
                print(f"[遥操作] 已恢复{pose_name}（速度/航向清零，obs 历史按上电逻辑重填）")

            # ---- 策略推理（100Hz）----
            if viewer is not None:
                commands = np.array([kb["vx"], kb["yaw_rate"], kb["height"]], dtype=np.float64)
            else:
                commands = commands_at(step * policy_dt)
            # 关键：mj_step 只积分不刷新派生量，cvel 停留在上一子步（5ms 前）。
            # 不刷新的话 mj_objectVelocity 读到的角速度（策略的陀螺仪观测）滞后 5ms，
            # 静态站立无感，加速瞬态的快俯仰动力学会因此欠阻尼而向后掀翻。
            mujoco.mj_forward(model, data)
            dof_pos, feedback_vel = read_dof_state()
            if h7_adapter is not None:
                dof_vel = feedback_vel
            base_quat_xyzw, base_ang_vel_body, base_lin_vel_body = read_base_state()
            # 与训练 chuanliantui 的 10ms 位置差分速度定义一致；仅供诊断。
            current_base_pos = data.qpos[base_qadr:base_qadr + 3].copy()
            velocity_pos_diff = quat_rotate_inverse(
                base_quat_xyzw, (current_base_pos - previous_base_pos) / policy_dt)
            previous_base_pos = current_base_pos
            source_valid = h7_adapter is None or (
                h7_adapter.valid and all(np.isfinite(value).all() for value in (
                    base_quat_xyzw, base_ang_vel_body, dof_pos, dof_vel, commands, last_action))
                and np.linalg.norm(base_quat_xyzw) > 1e-6)
            obs = (build_obs(base_quat_xyzw, base_ang_vel_body, dof_pos, dof_vel, commands, last_action)
                   if source_valid else np.zeros(NUM_OBS, dtype=np.float64))

            if h7_history is not None:
                # 模拟 H7 已投入 RL：十拍预热；无效观测清历史、重新预热。
                # H7 不使用训练端首次轮接地门控。
                obs_history, policy_active = h7_history.update(obs, source_valid)
                h7_output_ready = h7_history.ready
            else:
                obs_history = np.concatenate([obs_history[NUM_OBS:], obs])
                newly_landed = args.standup and not has_landed and has_wheel_contact()
                if newly_landed:
                    has_landed = True
                    print(f"[{step:5d}] 首次轮接地；下一控制步开始策略推理。")
                policy_active = not args.standup or (has_landed and not newly_landed)
            if policy_active:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                    hist_t = torch.from_numpy(obs_history).float().unsqueeze(0)
                    action_t, latent_t = policy.act_inference(obs_t, hist_t)
                    if args.diagnostic_true_vx and step * policy_dt >= args.diagnostic_true_vx_after:
                        # 单因素消融：只用仿真真值替换 actor 的 x 速度 latent。
                        # 保留原 encoder 输出用于记录，其余 latent/观测保持不变。
                        actor_latent = latent_t.clone()
                        actor_latent[:, 0] = float(velocity_pos_diff[0] * COMMANDS_SCALE[0])
                        action_t = policy.actor(torch.cat((obs_t, actor_latent), dim=-1))
                action = action_t.squeeze(0).cpu().numpy().astype(np.float64)
                latent = latent_t.squeeze(0).cpu().numpy().astype(np.float64)
                if h7_adapter is not None:
                    if not (np.isfinite(action).all() and np.isfinite(latent).all()):
                        action = np.zeros(NUM_ACTIONS, dtype=np.float64)
                        latent = np.zeros(LATENT_DIM, dtype=np.float64)
                        h7_output_ready = False
                        policy_active = False
                    else:
                        action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
            else:
                action = np.zeros(NUM_ACTIONS, dtype=np.float64)
                latent = np.zeros(LATENT_DIM, dtype=np.float64)
            if args.diagnostic_report:
                virtual_torque, motor_torque = control_outputs(action)
                # tendon motor 属于 actuator 力，不在 qfrc_passive 中；只累加两侧弹簧。
                gas_spring_qfrc = (
                    data.actuator_force[gas_spring_actuator_ids]
                    @ data.actuator_moment[gas_spring_actuator_ids]
                )
                contact_rows = []
                contact_force = np.zeros(6)
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    if contact.geom1 != floor_gid and contact.geom2 != floor_gid:
                        continue
                    mujoco.mj_contactForce(model, data, contact_id, contact_force)
                    contact_rows.append({"pos": contact.pos.tolist(), "normal_force": float(contact_force[0]),
                                         "geom1": int(contact.geom1), "geom2": int(contact.geom2)})
                diagnostic_rows.append({
                    "t": step * policy_dt, "active": policy_active,
                    "base_pos": current_base_pos.tolist(), "base_quat_xyzw": base_quat_xyzw.tolist(),
                    "pitch_deg": math.degrees(math.asin(float(np.clip(
                        quat_rotate_inverse(base_quat_xyzw, np.array([0., 0., -1.]))[0], -1., 1.)))),
                    "velocity_instant": base_lin_vel_body.tolist(),
                    "velocity_pos_diff": velocity_pos_diff.tolist(), "estimated_velocity": (latent / 2.).tolist(),
                    "commands": commands.tolist(), "obs": obs.tolist(), "history": obs_history.tolist(),
                    "action": action.tolist(), "dof_pos": dof_pos.tolist(), "dof_vel": dof_vel.tolist(),
                    "torque_at_policy_start": virtual_torque.tolist(),
                    "motor_torque_at_policy_start": motor_torque.tolist(),
                    "gas_spring_actuator_force_n": data.actuator_force[gas_spring_actuator_ids].tolist(),
                    "gas_spring_knee_torque_nm": gas_spring_qfrc[dof_adr[[1, 4]]].tolist(),
                    "physical_dof_pos": data.qpos[qpos_adr].tolist(),
                    "physical_dof_vel": data.qvel[dof_adr].tolist(),
                    "closed_pin_residual_m": ({side[0]: adapter._pin_residual(side).tolist()
                                                for side in adapter._SIDES} if adapter is not None else None),
                    "scheduled_push_force_world_n": (push_schedule.force_at(step * policy_dt).tolist()
                                                       if push_schedule else [0., 0., 0.]),
                    "h7_geometry_valid": bool(h7_adapter.valid) if h7_adapter is not None else None,
                    "h7_output_ready": bool(h7_output_ready) if h7_adapter is not None else None,
                    "motor_pos": data.qpos[model.jnt_qposadr[model.actuator_trnid[
                        adapter.actuator_ids if adapter is not None else actuator_ids, 0]]].tolist(),
                    "motor_vel": data.qvel[model.jnt_dofadr[model.actuator_trnid[
                        adapter.actuator_ids if adapter is not None else actuator_ids, 0]]].tolist(),
                    "passive_torque": data.qfrc_passive[dof_adr].tolist(),
                    "whole_robot_com": data.subtree_com[base_bid].tolist(),
                    "wheel_body_positions": [data.xpos[bid].tolist() for bid in sorted(wheel_bids)],
                    "floor_contacts": contact_rows,
                })
            if args.wheel_vel_limit is not None:
                limited_wheels = np.abs(action[[2, 5]] * VEL_ACTION_SCALE) > args.wheel_vel_limit
                wheel_limit_steps += bool(np.any(limited_wheels))
                wheel_limit_targets += int(np.count_nonzero(limited_wheels))
            # latent 前 3 维被训练监督为 base_lin_vel*2.0（ppo.py:265，lin_vel scale=2），
            # chuanliantui 前向为 +x，因此 latent[0]/2 是策略内部估计的前向速度。

            # ---- PD 内环（500Hz）：算力矩 -> 步进 -> 刷新 pos/vel ----
            # 训练每子步是 simulate 后 compute_dof_vel（legged_robot.py:111-115），最后一个
            # 子步结束时 dof_vel 是最新后向差分；闭链的虚拟膝速度用电机速度反算。
            # 若在步进前读取，obs 里的速度会滞后一个子步 2ms。
            for substep in range(DECIMATION):
                torque, motor_torque = control_outputs(action)
                if adapter is not None:
                    data.ctrl[adapter.actuator_ids] = motor_torque
                else:
                    data.ctrl[actuator_ids] = torque
                data.ctrl[gas_spring_actuator_ids] = args.gas_spring_force
                if push_schedule:
                    force = push_schedule.force_at((step * DECIMATION + substep) * SIM_DT)
                    data.xfrc_applied[base_bid, :] = 0.
                    data.xfrc_applied[base_bid, :3] = force
                    push_impulse += force * SIM_DT
                mujoco.mj_step(model, data)
                dof_pos, feedback_vel = read_dof_state()
                if h7_adapter is not None:
                    dof_vel = feedback_vel
                else:
                    dof_vel = wrap_to_pi(dof_pos - last_dof_pos) / SIM_DT
                    if feedback_vel is not None:
                        dof_vel[1], dof_vel[4] = feedback_vel
                last_dof_pos = dof_pos.copy()

            last_action = action

            if viewer is not None:
                viewer.sync()
                # 实时节流：让仿真按真实时间播放，便于观察（--realtime，默认开）
                if args.realtime:
                    target = wall_start + (step + 1) * policy_dt
                    sleep_t = target - time.perf_counter()
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            if step % args.log_every == 0:
                base_z = data.qpos[base_qadr + 2]
                base_x = data.qpos[base_qadr]
                # 双源速度交叉验证：机体系(objectVelocity) vs 世界系(qvel)；yaw≈0 时两者应接近
                v_fwd = base_lin_vel_body[0]
                vx_world = data.qvel[base_vadr]
                # 姿态（重力投影）与轮速：判断是否在前倾、轮子是否打滑（轮速*0.0625 应≈车速）
                gravity_world = np.array([0.0, 0.0, -1.0])
                pg = quat_rotate_inverse(base_quat_xyzw, gravity_world)
                wl = dof_vel[2] * 0.0625
                wr = dof_vel[5] * 0.0625
                print(f"[{step:5d}] x={base_x:+.3f} z={base_z:.3f} v_fwd={v_fwd:+.3f} "
                      f"v̂={latent[0]/2.0:+.3f} vx_w={vx_world:+.3f} "
                      f"轮速l/r={wl:+.2f}/{wr:+.2f}m/s "
                      f"pg_xyz=[{pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f}] yaw令={commands[1]:+.2f} "
                      f"|a|max={np.abs(action).max():.3f}")

            step += 1
    except KeyboardInterrupt:
        print("收到 Ctrl+C，退出仿真。")
    finally:
        if viewer is not None:
            viewer.close()

    print(f"仿真结束：{step} 个策略步。")
    if args.diagnostic_report:
        with Path(args.diagnostic_report).open("x") as stream:
            json.dump({"metadata": {
                "checkpoint": str(Path(args.checkpoint).resolve()), "model_xml": str(Path(model_xml).resolve()),
                "true_vx_override": args.diagnostic_true_vx, "velocity_definition": "10ms position difference in current body frame",
                "true_vx_after_s": args.diagnostic_true_vx_after,
                "sim_dt": SIM_DT, "policy_dt": policy_dt, "friction_override": args.friction,
                "dof_damping": model.dof_damping[dof_adr].tolist(),
                "dof_frictionloss": model.dof_frictionloss[dof_adr].tolist(),
                "dof_armature": model.dof_armature[dof_adr].tolist(),
                "joint_limit_solref": model.jnt_solref[joint_ids].tolist(),
                "joint_limit_solimp": model.jnt_solimp[joint_ids].tolist(),
                "joint_range": model.jnt_range[joint_ids].tolist(),
                "standup": args.standup, "closed_chain": args.closed_chain,
                "gas_spring_force_n_per_side": args.gas_spring_force,
                "push_schedule": push_schedule.to_dict() if push_schedule else None,
                "scheduled_push_impulse_world_ns": push_impulse.tolist(),
                "scheduled_push_force_sampling": "策略步起点的计划力；冲量按实际提交的每2ms力累计",
                "gas_spring_actuator_names": GAS_SPRING_ACTUATOR_NAMES,
                "gas_spring_knee_joint_names": ["rf1", "lf1"],
                "closed_chain_controller": args.closed_chain_controller if args.closed_chain else None,
                "h7_rear_zero_override": args.h7_rear_zero,
                "h7_front_zero": h7_adapter.front_zero.tolist() if h7_adapter is not None else None,
                "h7_rear_zero": h7_adapter.rear_zero.tolist() if h7_adapter is not None else None,
                "physical_dof_names": JOINT_NAMES,
                "motor_names": [model.joint(model.actuator_trnid[aid, 0]).name for aid in (
                    adapter.actuator_ids if adapter is not None else actuator_ids)],
                "wheel_vel_limit": args.wheel_vel_limit,
            }, "samples": diagnostic_rows}, stream, allow_nan=False)
        print(f"诊断轨迹（均在策略推理时刻采样）: {args.diagnostic_report}")
    if args.wheel_vel_limit is not None:
        print(f"轮目标限幅触发：{wheel_limit_steps} 个策略步，"
              f"{wheel_limit_targets} 个轮目标（两轮合计）。")


def selfcheck(args):
    """策略等价性/形状自检：喂零输入，确认 act_inference 输出 (1,6)。"""
    policy = load_policy(args.checkpoint, device="cpu")
    obs = torch.zeros(1, NUM_OBS)
    hist = torch.zeros(1, NUM_ENCODER_OBS)
    with torch.no_grad():
        action, latent = policy.act_inference(obs, hist)
    print(f"自检通过：action.shape={tuple(action.shape)} latent.shape={tuple(latent.shape)}")
    assert action.shape == (1, NUM_ACTIONS), action.shape
    assert latent.shape == (1, LATENT_DIM), latent.shape
    print(f"零输入下 action = {action.squeeze(0).numpy()}")


def main():
    p = argparse.ArgumentParser(description="chuanliantui MuJoCo sim2sim 部署验证")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="25 维 chuanliantui 的完整 model_*.pt 路径；历史 27 维权重不兼容",
    )
    p.add_argument(
        "--model_xml",
        default=None,
        help="覆盖模型路径；未指定时默认串联代理，配合 --closed_chain 时默认 chuanliantui.xml",
    )
    p.add_argument(
        "--closed_chain",
        action="store_true",
        help="启用 CAD 闭链机构，默认使用 H7 解算/控制；默认关闭；两种机构均含气弹簧",
    )
    p.add_argument("--closed_chain_controller", choices=("h7", "cad"), default=None,
                   help="闭链控制：h7=H7 五连杆解算/PD/预热（默认），cad=原 CAD 多环解算")
    p.add_argument("--h7_rear_zero", type=float, nargs=2, metavar=("LF", "RF"), default=None,
                   help="H7 路径：CAD 后轴 q=0 对应的固件 qb [rad]；默认取 CAD 杆方向，仅为名义注册")
    p.add_argument(
        "--gas_spring_force",
        type=float,
        default=DEFAULT_GAS_SPRING_FORCE,
        help="串联代理/闭链每侧气弹簧恒定伸张推力 [N]，范围 0~150；默认 150 N，0 复现无弹簧条件",
    )
    p.add_argument("--render", action="store_true",
                   help="启动 MuJoCo passive viewer 和键盘遥操作；关闭窗口退出")
    p.add_argument("--gas_spring_view", action="store_true",
                   help="打开气弹簧特写：半透明机构、彩色端点与轴线；自动启用 --closed_chain --render")
    p.add_argument("--cmd_vx", type=float, default=0.0, help="目标前向线速度 [m/s]")
    p.add_argument("--cmd_yaw", type=float, default=0.0,
                   help="目标偏航角速度 [rad/s]；训练使用 heading_command=False，直接写入命令通道 1")
    p.add_argument("--cmd_height", type=float, default=None,
                   help="目标机身高度 [m]；H7 闭链默认 0.20，其余路径默认 0.32")
    p.add_argument("--cmd_delay", type=float, default=3.0,
                   help="速度/偏航命令延迟生效时间 [s]（先站稳再动，高度命令不受影响）")
    p.add_argument("--cmd_ramp", type=float, default=1.0,
                   help="速度/偏航命令线性爬升时长 [s]（0=阶跃）")
    p.add_argument("--friction", type=float, default=None,
                   help="覆盖所有 geom 的滑动摩擦系数（默认用 XML 里的 0.5；训练等效均值约 0.75）")
    p.add_argument("--init_height", type=float, default=0.33, help="串联代理复位时的初始基座高度 [m]")
    p.add_argument("--initial_dof_pos", type=float, nargs=NUM_ACTIONS,
                   default=DEFAULT_DOF_POS.tolist(), metavar=("RF0", "RF1", "RWHEEL", "LF0", "LF1", "LWHEEL"),
                   help="复位训练 DOF [rf0, rf1, rfwheel, lf0, lf1, lfwheel] [rad]；"
                        "不用 --standup 时，策略从第一个控制步开始接管")
    p.add_argument("--ground_start", action="store_true",
                   help="按当前 initial_dof_pos 的碰撞网格最低点自动贴地；不能与 --standup 同用")
    p.add_argument("--ground_clearance", type=float, default=0.002,
                   help="--ground_start 的碰撞网格最低点离地间隙 [m]")
    p.add_argument("--standup", action="store_true",
                   help="使用当前起立训练的初态：0.15 m 地面后摆；覆盖 --init_height 和 --initial_dof_pos，首次轮接地后的下一控制步才推理策略")
    p.add_argument("--log_every", type=int, default=100, help="每多少策略步打印一次")
    p.add_argument("--wheel_vel_limit", type=float, default=None,
                   help="轮目标速度绝对限幅 [rad/s]；H7 闭链固定 20，其余路径默认不限制")
    p.add_argument("--sim_time", type=float, default=None,
                   help="运行的仿真时长 [s]；默认持续运行直到关闭窗口或 Ctrl+C")
    p.add_argument("--selfcheck", action="store_true", help="仅做策略形状自检，不跑仿真")
    p.add_argument("--h7_no_wheel_target_clip", action="store_true",
                   help="显式回放已取消轮速目标裁剪的H7变体；仍保留原电机力矩限幅")
    p.add_argument("--push_schedule", default=None,
                   help="加载固定JSON推力计划，在base_link质心每2ms施加世界系水平力")
    p.add_argument("--diagnostic_report", default=None,
                   help="无渲染有限时长诊断：将每个策略时刻的状态/估计/动作写入新 JSON 文件")
    p.add_argument("--diagnostic_true_vx", action="store_true",
                   help="仅诊断：用与训练同定义的仿真真 vx 替换 actor 的前向速度 latent，不用于部署")
    p.add_argument("--diagnostic_true_vx_after", type=float, default=0.0,
                   help="诊断替换起始时刻 [s]；设为 5 可保留原始起立轨迹")
    p.add_argument("--no_realtime", dest="realtime", action="store_false",
                   help="关闭实时节流（默认按真实时间播放，便于观察）")
    p.set_defaults(realtime=True)
    args = p.parse_args()
    if args.gas_spring_view:
        args.closed_chain = True
        args.render = True
    if not args.closed_chain and (args.closed_chain_controller is not None or args.h7_rear_zero is not None):
        p.error("闭链控制参数需要 --closed_chain")
    args.closed_chain_controller = args.closed_chain_controller or "h7"
    is_h7 = args.closed_chain and args.closed_chain_controller == "h7"
    if args.h7_rear_zero is not None and (
        not is_h7 or not np.isfinite(args.h7_rear_zero).all()
    ):
        p.error("--h7_rear_zero 需要 H7 闭链路径和两个有限角度")
    if args.cmd_height is None:
        args.cmd_height = 0.20 if is_h7 else 0.32
    if args.h7_no_wheel_target_clip and (not is_h7 or args.wheel_vel_limit is not None):
        p.error("--h7_no_wheel_target_clip 仅用于H7闭链，且不能同时指定 --wheel_vel_limit")
    if is_h7 and not args.h7_no_wheel_target_clip:
        if args.wheel_vel_limit is not None and args.wheel_vel_limit != 20.0:
            p.error("H7 闭链固定轮目标限幅 20 rad/s；其他限幅对照请使用 cad 控制路径")
        args.wheel_vel_limit = 20.0
    if not 0.0 <= args.gas_spring_force <= DEFAULT_GAS_SPRING_FORCE:
        p.error("--gas_spring_force 必须在 0~150 N 内（受 MJCF actuator ctrlrange 限制）")
    if args.wheel_vel_limit is not None and (
        not math.isfinite(args.wheel_vel_limit) or args.wheel_vel_limit <= 0.0
    ):
        p.error("--wheel_vel_limit 必须是正的有限数值")
    if args.sim_time is not None and (not math.isfinite(args.sim_time) or args.sim_time <= 0.0):
        p.error("--sim_time 必须是正的有限数值")
    if args.log_every <= 0:
        p.error("--log_every 必须为正整数")
    if args.diagnostic_true_vx and not args.diagnostic_report:
        p.error("--diagnostic_true_vx 需要 --diagnostic_report 留存消融证据")
    if not math.isfinite(args.diagnostic_true_vx_after) or args.diagnostic_true_vx_after < 0:
        p.error("--diagnostic_true_vx_after 必须是非负有限数值")
    if args.diagnostic_report:
        if args.render or args.sim_time is None or args.selfcheck:
            p.error("--diagnostic_report 需要无渲染的有限 --sim_time 行为回放")
        if Path(args.diagnostic_report).exists():
            p.error("不覆盖已有诊断报告")

    if args.selfcheck:
        selfcheck(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
