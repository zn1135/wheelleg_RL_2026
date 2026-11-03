"""Independent mjlab environment for the H7 LQR teacher (no policy/runner)."""
from dataclasses import dataclass
import math
import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from ..envs.chuanliantui.huananhu_env_cfg import make_huananhu_env_cfg
from ..envs.chuanliantui.huananhu_action import wheel_response_input
from ..envs.chuanliantui.gas_spring import gas_spring_force
from ..envs.chuanliantui.model import (ACTUATED_JOINTS, GAS_SPRING_JOINTS, MODEL_XML,
                                      gas_spring_actuator_ids, wheel_response_parameters)
from .controller import LqrVmcController
from .mechanics import DT


def body_angles(quaternion):
    w, x, y, z = quaternion.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    upright = 1-2*(x*x+y*y)
    return pitch, roll, yaw, upright


@dataclass(kw_only=True)
class TeacherActionCfg(ActionTermCfg):
    initial: str = 'standing'
    velocity_integral: bool = False
    ground_start_hold_s: float = 0.

    def build(self, env):
        return TeacherAction(self, env)


class TeacherAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.controller = LqrVmcController(self.num_envs, self.device,velocity_integral=cfg.velocity_integral)
        if not math.isfinite(cfg.ground_start_hold_s) or cfg.ground_start_hold_s < 0.:
            raise ValueError('ground_start_hold_s must be finite and nonnegative')
        self._hold_ticks = math.ceil(cfg.ground_start_hold_s/DT-1e-9) if cfg.initial == 'ground' else 0
        self._control_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.control_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.motor_ids = self._entity.find_joints(ACTUATED_JOINTS, preserve_order=True)[0]
        self.qadr = self._entity.indexing.joint_q_adr[self.motor_ids]
        self.vadr = self._entity.indexing.joint_v_adr[self.motor_ids]
        spring_joints = self._entity.find_joints(GAS_SPRING_JOINTS, preserve_order=True)[0]
        self.sqadr = self._entity.indexing.joint_q_adr[spring_joints]
        self.svadr = self._entity.indexing.joint_v_adr[spring_joints]
        self.spring_ids = torch.as_tensor(gas_spring_actuator_ids(env.sim.mj_model), device=self.device)
        actuator_ids = [
            env.sim.mj_model.actuator('robot/' + name + '_motor').id for name in ACTUATED_JOINTS
        ]
        self.motor_actuators = torch.tensor(actuator_ids, device=self.device)
        self.response = torch.tensor(wheel_response_parameters(env.sim.mj_model), device=self.device,
                                     dtype=torch.float32)
        self.command = torch.zeros(self.num_envs, 3, device=self.device)
        self.command[:, 2] = .15
        self.torques = torch.zeros(self.num_envs, 6, device=self.device)
        self.h7_torques = torch.zeros_like(self.torques)
        self.previous_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.velocity_ready = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # H7内部四腿两轮顺序，与现有RL配对映射完全一致。
        self.order = [0, 1, 3, 4, 2, 5]
        self.sign = torch.tensor([1., 1., -1., -1., 1., -1.], device=self.device)
        self.limits = torch.tensor(env.sim.mj_model.actuator_ctrlrange[actuator_ids, 1],
                                   dtype=torch.float32, device=self.device)

    @property
    def action_dim(self):
        return 3

    @property
    def raw_action(self):
        return self.command

    def process_actions(self, actions):
        self.command.copy_(actions)

    def apply_actions(self):
        self.compute_torques()
        effort = self.torques.clone()
        motor_velocity = self._env.sim.data.qvel[:, self.vadr]
        effort[:, [2, 5]] = wheel_response_input(effort[:, [2, 5]], motor_velocity[:, [2, 5]], self.response)
        self._entity.set_joint_effort_target(effort, joint_ids=self.motor_ids)
        self._env.sim.data.ctrl[:, self.spring_ids] = gas_spring_force(
            self._env.sim.data.qpos[:, self.sqadr], self._env.sim.data.qvel[:, self.svadr],
            torch.zeros_like(self._env.sim.data.qpos[:, self.sqadr]))

    def compute_torques(self, *, dt=DT, position_reference=None):
        """Read live feedback and compute requests without writing actuators."""
        hold = self._control_ticks < self._hold_ticks
        self.control_active.copy_(~hold)
        robot = self._entity.data
        q = self._env.sim.data.qpos[:, self.qadr][:, self.order] * self.sign
        dq = self._env.sim.data.qvel[:, self.vadr][:, self.order] * self.sign
        q = q[:, :4].reshape(-1, 2, 2).clone()
        q[:, :, 0] += math.pi  # H7 task_comm.c只给前髋加π。
        pitch, roll, yaw, upright = body_angles(robot.root_link_quat_w)
        velocity = robot.root_link_lin_vel_w
        acc_w = torch.where(self.velocity_ready[:, None], (velocity-self.previous_velocity)/dt, 0.)
        self.previous_velocity.copy_(velocity)
        self.velocity_ready.fill_(True)
        acceleration = acc_w[:, 0]*yaw.cos() + acc_w[:, 1]*yaw.sin()
        feedback = self._env.sim.data.actuator_force[:, self.motor_actuators][:, self.order] * self.sign
        output = self.controller.step(
            q, dq[:, :4].reshape(-1, 2, 2), dq[:, 4:], pitch, roll, yaw,
            robot.root_link_ang_vel_b[:, 1], robot.root_link_ang_vel_b[:, 2], upright,
            acceleration, acc_w[:, 2], feedback[:, :4].reshape(-1, 2, 2), self.command,
            position_reference=position_reference)
        self.h7_torques.copy_(output)
        self.torques[:, self.order] = output * self.sign
        # XML现有执行器限幅仍保留；H7请求与实际仿真请求分开记录。
        self.torques.copy_(torch.clamp(self.torques, -self.limits, self.limits))
        self.torques[hold] = 0.
        self.h7_torques[hold] = 0.
        # 等待期间只保留被动物理；接管时从真实落地状态重新开始控制历史。
        if hold.any():
            self.controller.reset(hold, standing=False)
            self.previous_velocity[hold] = 0.
            self.velocity_ready[hold] = False
        self._control_ticks.add_(round(dt/DT))
        return self.torques

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.controller.reset(ids, standing=self.cfg.initial == 'standing')
        self.torques[ids] = 0
        self.h7_torques[ids] = 0
        self.previous_velocity[ids] = 0
        self.velocity_ready[ids] = False
        self._control_ticks[ids] = 0
        self.control_active[ids] = self._hold_ticks == 0
        self.command[ids] = self.command.new_tensor([0., 0., .15])


def reset_pose(env, env_ids, *, initial):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene['robot']
    joint = robot.data.default_joint_pos[ids].clone()
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] = env.scene.env_origins[ids]
    root[:, 3:7] = root.new_tensor([1., 0., 0., 0.])
    root[:, 7:] = 0
    if initial == 'ground':
        from ..envs.chuanliantui.huananhu_poses import POSE_JOINT_NAMES
        with np.load(MODEL_XML.with_name('huananhu_pose_bank.npz')) as bank:
            # 正常趴地采用同分支双腿，绝不加入倒置根姿态。
            left = torch.as_tensor(bank['negative_left_pos'], device=env.device)
            right = torch.as_tensor(bank['negative_right_pos'], device=env.device)
        rows = torch.randint(len(left), (len(ids),), device=env.device)
        poses = torch.cat((left[rows], right[rows]), -1)
        order = [POSE_JOINT_NAMES.index(name) for name in robot.joint_names]
        joint.copy_(poses[:, order])
        root[:, 2] += .25
    else:
        root[:, 2] += .38
    robot.write_joint_state_to_sim(joint, torch.zeros_like(joint), env_ids=ids)
    robot.write_root_state_to_sim(root, env_ids=ids)


def observe(env):
    return env.scene['robot'].data.root_link_lin_vel_b


def make_teacher_env_cfg(*, initial='standing', num_envs=1, seed=42, velocity_integral=False,
                         ground_start_hold_s=0.):
    cfg = make_huananhu_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.mujoco.timestep = DT
    cfg.decimation = 1
    cfg.actions = {'teacher': TeacherActionCfg(entity_name='robot', initial=initial,
        velocity_integral=velocity_integral, ground_start_hold_s=ground_start_hold_s)}
    cfg.observations = {'actor': ObservationGroupCfg(
        terms={'velocity': ObservationTermCfg(func=observe)}, concatenate_terms=True,
        enable_corruption=False, nan_policy='error')}
    cfg.commands = {}
    cfg.events = {'reset_pose': EventTermCfg(func=reset_pose, mode='reset', params={'initial': initial})}
    cfg.metrics = {}
    cfg.rewards = {}
    cfg.terminations = {}  # 单次回放不自动重开回合，失败由控制器锁停并记录。
    cfg.curriculum = {}
    return cfg
