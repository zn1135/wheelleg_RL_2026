"""CPU 气弹簧数值/API契约检查；不创建 Isaac 仿真、不替代行为验收。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import isaacgym  # 必须先于 torch
from isaacgym import gymapi, gymtorch
import mujoco
import numpy as np
import torch

from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.envs.chuanliantui.chuanliantui import Chuanliantui
from wheel_legged_gym.envs.chuanliantui.chuanliantui_config import ChuanliantuiCfg
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup import ChuanliantuiStandup
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup_config import ChuanliantuiStandupCfg
from wheel_legged_gym.utils.chuanliantui_gas_spring import GasSpringGeometry


ROOT = Path(__file__).resolve().parents[2]
DOF_NAMES = ("rf0", "rf1", "rfwheel", "lf0", "lf1", "lfwheel")
SIDES = ("right", "left")


class TendonOracle:
    """用实际 MJCF 的 tendon 广义力和长度差分作为独立参照。"""

    def __init__(self, filename):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "sim2sim" / filename))
        self.data = mujoco.MjData(self.model)
        self.gas_ids = [self.model.actuator(side + "_gas_spring_motor").id for side in SIDES]
        self.tendon_ids = [self.model.tendon(side + "_gas_spring_tendon").id for side in SIDES]
        self.joints = [self.model.joint(name).id for name in DOF_NAMES]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]

    def set_pose(self, q, root_quat, force):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0., 0., 1.)
        self.data.qpos[3:7] = root_quat  # MuJoCo wxyz
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.gas_ids] = force
        # 不步进/求闭链平衡：只比较给定主链姿态的弹簧广义力。
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_actuator[self.vadr].copy()

    def finite_difference(self, force):
        values = []
        for tendon_id, qadr in zip(self.tendon_ids, self.qadr[[1, 4]]):
            initial = self.data.qpos[qadr]
            eps = 1e-6
            self.data.qpos[qadr] = initial + eps
            mujoco.mj_forward(self.model, self.data)
            plus = self.data.ten_length[tendon_id]
            self.data.qpos[qadr] = initial - eps
            mujoco.mj_forward(self.model, self.data)
            minus = self.data.ten_length[tendon_id]
            self.data.qpos[qadr] = initial
            values.append(force * (plus - minus) / (2 * eps))
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(values)


def pose_grid():
    """覆盖硬限位端点、常用姿态、后摆、跨wrap及非对称两膝。"""
    quats = np.asarray(((1., 0., 0., 0.), (.8, .2, -.3, .4), (.1, -.6, .7, .2)))
    quats /= np.linalg.norm(quats, axis=1)[:, None]
    poses = []
    for index, (hip, knee) in enumerate(
            (h, k) for h in (-3.3, -1.8, -.06, .7, 3.3) for k in (-.12, 0., .1, .63, .77)):
        poses.append((np.array((hip, knee, 0., -hip, -knee, 0.)), quats[index % 3]))
    poses.extend((
        (np.array((-.54, .69, 0., .50, -.63, 0.)), quats[1]),
        (np.array((1.7, -.12, 0., -.8, -.77, 0.)), quats[2]),
    ))
    return poses


def geometry(dtype):
    cfg = ChuanliantuiCfg()
    path = cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=str(ROOT))
    return GasSpringGeometry(path, device="cpu", dtype=dtype)


def check_tendon_geometry():
    oracles = [TendonOracle(name) for name in ("chuanliantui.xml", "chuanliantui_train_proxy.xml")]
    poses = pose_grid()
    knees = np.asarray([q[[1, 4]] for q, _ in poses])
    errors = []
    for force in (0., 75., 150.):
        reference = []
        for q, quat in poses:
            torques = [oracle.set_pose(q, quat, force) for oracle in oracles]
            np.testing.assert_allclose(torques[0], torques[1], rtol=0, atol=1e-10)
            for oracle, torque in zip(oracles, torques):
                np.testing.assert_allclose(torque[[0, 2, 3, 5]], 0., rtol=0, atol=1e-10)
                np.testing.assert_allclose(oracle.data.qfrc_actuator[:6], 0., rtol=0, atol=1e-10)
                if force == 150.:
                    np.testing.assert_allclose(torque[[1, 4]], oracle.finite_difference(force),
                                               rtol=0, atol=5e-8)
            reference.append(torques[0][[1, 4]])
        reference = np.asarray(reference)
        for dtype, atol in ((torch.float32, 5e-6), (torch.float64, 1e-10)):
            actual = geometry(dtype).knee_torques(torch.tensor(knees, dtype=dtype), force).numpy()
            np.testing.assert_allclose(actual, reference, rtol=0, atol=atol)
            errors.append(float(np.max(np.abs(actual - reference))))
            if force == 0.:
                assert np.count_nonzero(actual) == 0, "0 N 出现被动力矩"
        # 前25组是镜像姿态，CAD两侧微小制造坐标舍入只容许数值级误差。
        np.testing.assert_allclose(reference[:25, 0], -reference[:25, 1], rtol=0, atol=1e-9)
    assert np.ptp(reference[:25, 0]) > .5, "检查姿态未覆盖气弹簧力臂变化"
    print("PASS: 27姿态×3力值，Torch float32/64 vs闭链/串联tendon，长度差分、镜像、0N；最大误差 {:.3g} Nm".format(max(errors)))


class ForceBoundary:
    """只替代 Isaac API，所有力矩计算、推力累加及step钩子用真实源码。"""

    body_indices = {"lf0": 1, "rf1": 2, "base_link": 3, "rf0": 5, "lf1": 6}

    def __init__(self, env, pending_quat):
        self.env = env
        self.pending_quat = pending_quat
        self.calls, self.motor_calls, self.events = [], [], []

    def find_actor_rigid_body_handle(self, env_handle, actor_handle, name):
        return self.body_indices[name]

    def refresh_actor_root_state_tensor(self, sim):
        self.env.root_states[:, 3:7] = self.pending_quat
        self.events.append("refresh_root")

    def apply_rigid_body_force_tensors(self, sim, forces, torques, space):
        assert space == gymapi.ENV_SPACE
        assert torch.isfinite(forces).all() and torch.isfinite(torques).all()
        self.calls.append((forces.clone(), torques.clone()))
        self.events.append("external")

    def set_dof_actuation_force_tensor(self, sim, torques):
        self.motor_calls.append(torques.clone())
        self.events.append("motor")

    def simulate(self, sim):
        self.events.append("simulate")

    def fetch_results(self, sim, wait):
        pass

    def refresh_dof_state_tensor(self, sim):
        pass


def make_env(force=150., push=True):
    poses = pose_grid()
    cfg = ChuanliantuiStandupCfg()
    cfg.gas_spring.force_n = force
    cfg.domain_rand.push_robots = push

    def initialize_without_physics(env, cfg, *args):
        env.cfg, env.device, env.dt = cfg, "cpu", .01
        env.num_envs, env.num_dofs, env.num_bodies = len(poses), 6, 8
        env.dof_names = list(DOF_NAMES)
        env.envs = env.actor_handles = list(range(env.num_envs))
        env.sim, env.sim_params = object(), SimpleNamespace(dt=.002)
        env.dof_pos = torch.tensor(np.asarray([q for q, _ in poses]), dtype=torch.float32)
        env.dof_vel = torch.zeros_like(env.dof_pos)
        env.root_states = torch.zeros(env.num_envs, 13)
        env.root_states[:, 6] = 1.
        # 故意保留陈旧策略拍姿态；refresh API 才注入本子步的姿态。
        env.base_quat = env.root_states[:, 3:7].clone()
        quats = np.asarray([quat for _, quat in poses])[:, [1, 2, 3, 0]]
        env.gym = ForceBoundary(env, torch.tensor(quats, dtype=torch.float32))
        env.rigid_body_external_forces = torch.zeros(env.num_envs, env.num_bodies, 3)
        env.rigid_body_external_torques = torch.zeros_like(env.rigid_body_external_forces)
        env.feet_indices = torch.tensor([0, 7])

    with patch.object(LeggedRobot, "__init__", initialize_without_physics):
        env = ChuanliantuiStandup(cfg, None, None, "cpu", True)
    assert hasattr(env, "gas_spring_geometry"), "Chuanliantui构造器未初始化气弹簧"
    return env


def submit(env):
    with patch.object(gymtorch, "unwrap_tensor", side_effect=lambda value: value):
        env._apply_external_forces()


def assert_body_torques_match_tendon(env, torques):
    oracle = TendonOracle("chuanliantui_train_proxy.xml")
    for index, (q, quat) in enumerate(pose_grid()):
        oracle.set_pose(q, quat, env.gas_spring_force_n)
        generalized = np.zeros(oracle.model.nv)
        for name, body_index in env.gym.body_indices.items():
            body_id = oracle.model.body(name).id
            mujoco.mj_applyFT(oracle.model, oracle.data, np.zeros(3),
                             torques[index, body_index].double().numpy(),
                             oracle.data.xpos[body_id], body_id, generalized)
        np.testing.assert_allclose(generalized, oracle.data.qfrc_actuator, rtol=0, atol=6e-6)
    torch.testing.assert_close(torques.sum(dim=1), torch.zeros(env.num_envs, 3), rtol=0, atol=1e-7)


def check_external_force_composition():
    env = make_env()
    env.standup_push_force[:] = torch.tensor((3., -4., 0.))
    # 预填垃圾；每子步只能有本次的力，不能累加上次弹簧/推力。
    env.rigid_body_external_forces[:] = 19.
    env.rigid_body_external_torques[:] = 23.
    submit(env)
    assert len(env.gym.calls) == 1 and env.gym.events == ["refresh_root", "external"]
    forces, torques = env.gym.calls[-1]
    expected = torch.zeros_like(forces)
    expected[:, env.standup_push_body_index] = env.standup_push_force
    torch.testing.assert_close(forces, expected, rtol=0, atol=0)
    assert_body_torques_match_tendon(env, torques)
    submit(env)
    torch.testing.assert_close(env.gym.calls[-1][1], torques, rtol=0, atol=0)

    env.cfg.domain_rand.push_robots = False
    submit(env)
    assert not env.gym.calls[-1][0].any(), "关闭push后残留外力"
    torch.testing.assert_close(env.gym.calls[-1][1], torques, rtol=0, atol=0)
    env.gas_spring_force_n = 0.
    submit(env)
    assert not env.gym.calls[-1][0].any() and not env.gym.calls[-1][1].any()
    assert not env.gas_spring_knee_torques.any(), "0 N 仍记录旧被动力矩"
    zero_env = make_env(force=0., push=True)
    zero_env.standup_push_force[:] = torch.tensor((1., 2., 0.))
    submit(zero_env)
    assert zero_env.gym.calls[-1][0].any() and not zero_env.gym.calls[-1][1].any()
    print("PASS: 真实构造器/子步hook，世界轴刷新、非零刚体索引、反向纯扭矩、推力单次合成及关闭清零")


def check_motor_limit_and_substeps():
    env = make_env(push=True)
    env.standup_push_force[:] = torch.tensor((3., 4., 0.))
    env.default_dof_pos = torch.zeros_like(env.dof_pos)
    env.p_gains = torch.tensor((10., 10., 0., 10., 10., 0.))
    env.d_gains = torch.tensor((1., 1., .1, 1., 1., .1))
    env.torque_limits = torch.tensor((40., 40., 3.9, 40., 40., 3.9))
    env.torques_scale = torch.ones_like(env.dof_pos) * .7
    env.torques = torch.zeros_like(env.dof_pos)
    actions = torch.tensor((100., -100., 100., -100., 100., -100.)).repeat(env.num_envs, 1)
    expected_motor = torch.sign(actions) * env.torque_limits
    actual_motor = env._compute_torques(actions)
    torch.testing.assert_close(actual_motor, expected_motor, rtol=0, atol=0)

    env.envs_steps_buf = torch.zeros(env.num_envs, dtype=torch.long)
    env.action_fifo = torch.zeros(env.num_envs, 1, 6)
    env.action_delay_idx = torch.zeros(env.num_envs, dtype=torch.long)
    env.obs_buf, env.obs_history = torch.zeros(env.num_envs, 25), torch.zeros(env.num_envs, 125)
    env.privileged_obs_buf = None
    env.rew_buf, env.reset_buf, env.extras = torch.zeros(env.num_envs), torch.zeros(env.num_envs), {}
    # 隔离策略调度/物理结果/奖励；执行真实 base.step 的完整5次提交路径。
    env.render = env.pre_physics_step = env.post_physics_step = env.compute_dof_vel = lambda: None
    with patch.object(gymtorch, "unwrap_tensor", side_effect=lambda value: value):
        env.step(actions)
    assert len(env.gym.calls) == len(env.gym.motor_calls) == env.cfg.control.decimation == 5
    assert env.gym.events == ["motor", "refresh_root", "external", "simulate"] * 5
    for motor, (forces, torques) in zip(env.gym.motor_calls, env.gym.calls):
        torch.testing.assert_close(motor, expected_motor, rtol=0, atol=0)
        assert forces.any() and torques.any(), "合成时漏掉推力或弹簧"
    assert_body_torques_match_tendon(env, env.gym.calls[-1][1])
    combined = actual_motor[:, [1, 4]] + env.gas_spring_knee_torques
    assert (combined.abs() > 40.).all(), "弹簧不能受40 Nm电机限幅/随机倍率影响"
    print("PASS: 真实base.step每2ms一次合成，电机40/3.9限幅保留，被动弹簧独立于电机倍率/限矩")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    for check in (check_tendon_geometry, check_external_force_composition, check_motor_limit_and_substeps):
        check()
    print("CPU static/tensor/API checks only; no Isaac physics, policy rollout or training.")
