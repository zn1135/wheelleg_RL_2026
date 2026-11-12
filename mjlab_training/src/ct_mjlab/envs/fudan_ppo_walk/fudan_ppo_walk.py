"""mjlab physics with a separate Fudan task loop, no legacy command or reward path."""
from dataclasses import dataclass, field, replace
import math
import re
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from ct_mjlab.envs.chuanliantui.huananhu_env_cfg import make_huananhu_env_cfg
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction, HuananhuPdActionCfg
from ct_mjlab.envs.chuanliantui.pd_action import ChuanliantuiPdAction
from .reset_geometry import ResetGeometry, nominal_reset_pose
from ct_mjlab.fudan_ppo_walk.contract import MOTORS, LEGS, RESET_CONTRACT, RESET_V2_CONTRACT
from .fudan_ppo_walk_config import FudanWalkCfg, policy_dt, episode_steps
from .disturbances import DisturbanceConfig, DisturbanceState, SENSOR_SLOTS, from_task_config
from . import commands as command_sampling
from . import posture_reward
from .wheel_motor import WheelMotorResponse
from . import wheel_physics

@dataclass(kw_only=True)
class PhysicalActionCfg(HuananhuPdActionCfg):
    class_type: type = field(default_factory=lambda: PhysicalAction)
    task_cfg: object = None
    reference_randomization: bool = True
    disturbances: DisturbanceConfig = field(default_factory=from_task_config)

class PhysicalAction(HuananhuPdAction):

    def __init__(self, cfg, env):
        if cfg.task_cfg is None:
            cfg = replace(cfg, task_cfg=FudanWalkCfg())
        super().__init__(
            replace(
                cfg,
                motor_randomization=False,
                randomize_delays=False,
                action_delay_s=0.0,
                nominal_leg_offset_range=(0.0, 0.0),
                ground_start_hold_s=0.0
            ),
            env
        )
        self.cfg = cfg
        task = cfg.task_cfg or FudanWalkCfg()
        if cfg.reference_randomization:
            self.kp_scale.uniform_(*task.domain_rand.kp_range)
            self.kd_scale.uniform_(*task.domain_rand.kd_range)
            self.motor_strength.uniform_(*task.domain_rand.motor_strength_range)
            self.nominal_offset.uniform_(*task.domain_rand.nominal_offset_range)
            self._nominal_pos.add_(self.nominal_offset)
        self.last_sample_q = env.sim.data.qpos[:, self._motor_q_adr].clone()
        self.sample_velocity = torch.zeros_like(self.last_sample_q)
        self.substep_rows = []
        self.record_substeps = False
        self.rebuild_reset_poses()
        self.disturbances = DisturbanceState(
            cfg.disturbances,
            self.num_envs,
            self.device,
            env.cfg.seed if env.cfg.seed is not None else 1,
            physics_dt=task.sim.dt
        )
        self.disturbances.reset(slice(None), self.sensor_frame())
        self._base_id = int(env.sim.mj_model.body('robot/base_link').id)
        self.push_force_scale = torch.zeros(self.num_envs, device=self.device)
        self.wheel_motor = WheelMotorResponse(self,task.sim)

    def sensor_frame(self):
        q = self._env.sim.data.qpos
        gravity = quat_apply_inverse(q[:, 3:7], q.new_tensor([0.0, 0.0, -1.0]).expand(self.num_envs, -1))
        return torch.cat(
            (
                self._env.sim.data.qvel[:, 3:6] * self.cfg.task_cfg.normalization.angular_velocity,
                gravity,
                (q[:, self._motor_q_adr] - self.nominal_positions)[:, LEGS],
                self.sample_velocity * self.cfg.task_cfg.normalization.joint_velocity
            ),
            -1
        )

    # 500Hz内环沿用原PD与执行器，只在配置开启时应用独立延迟/推扰。
    def apply_actions(self):
        if self.cfg.disturbances.action_delay:
            self._control_actions.copy_(self.disturbances.action(self._raw_actions))
        super().apply_actions()
        if self.wheel_motor.enabled:
            wheels=self._wheel_action_ids
            ctrl=self.wheel_motor.apply(self._motor_response_torque[:,wheels],
                self._env.sim.data.qvel[:,self._motor_v_adr[wheels]])
            self._entity.set_joint_effort_target(ctrl,joint_ids=self._physical_joint_ids[wheels])
        if self.cfg.disturbances.pushes or self.disturbances.has_episode_wrench:
            quat = self._env.sim.data.qpos[:, 3:7]
            wrench = self.disturbances.wrench(quat)
            if self.cfg.disturbances.pushes:
                wrench[:, :3] += self.disturbances.force(quat, self.push_force_scale)
            self._env.sim.data.xfrc_applied[:, self._base_id] = wrench

    def rebuild_reset_poses(self):
        solver = ResetGeometry()
        poses = [solver.solve(row) for row in self.nominal_positions.detach().cpu().numpy()]
        names = self._entity.joint_names
        self.reset_joint_positions = torch.tensor(
            [[p['joint_positions'][name] for name in names] for p in poses],
            dtype=self._nominal_pos.dtype,
            device=self.device
        )
        self.reset_root_height = torch.tensor([p['height'] for p in poses], dtype=self._nominal_pos.dtype, device=self.device)

    def _ground_hold(self):
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions):
        self._raw_actions.copy_(actions.clamp(-self.cfg.task_cfg.normalization.clip, self.cfg.task_cfg.normalization.clip))
        self._control_actions.copy_(self._raw_actions)
        self.substep_rows.clear()

    def reset(self, env_ids=None):
        ChuanliantuiPdAction.reset(self, env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._motor_response_torque[ids] = 0.0
        if hasattr(self,'wheel_motor'):self.wheel_motor.reset(ids)
        self.action_delay_steps[ids] = 0
        self.wheel_action_delay_steps[ids] = 0
        if hasattr(self, 'last_sample_q'):
            self.last_sample_q[ids] = self._env.sim.data.qpos[ids][:, self._motor_q_adr]
            self.sample_velocity[ids] = 0.0
        if hasattr(self, 'disturbances'):
            self.disturbances.reset(ids, self.sensor_frame())
            if self.cfg.disturbances.pushes or self.disturbances.has_episode_wrench:
                self._env.sim.data.xfrc_applied[ids, self._base_id] = 0.0

def current_sensors(env):
    return env.scene['robot'].data.projected_gravity_b

def sample_substep(env):
    action = env.action_manager.get_term('pd')
    q = env.sim.data.qpos[:, action._motor_q_adr]
    action.sample_velocity.copy_((q - action.last_sample_q + math.pi).remainder(2 * math.pi).sub(math.pi) / env.physics_dt)
    action.last_sample_q.copy_(q)
    if action.cfg.disturbances.sensor_delay:
        action.disturbances.sample(action.sensor_frame())
    if action.record_substeps:
        from ct_mjlab.fudan_ppo_walk.measurement import gpu_row
        action.substep_rows.append(gpu_row(env).clone())
    return action.sample_velocity.abs().amax(-1)

def reset_pose(env, env_ids):
    robot = env.scene['robot']
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    root = robot.data.default_root_state[env_ids].clone()
    action = env.action_manager.get_term('pd')
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, 2] += action.reset_root_height[env_ids]
    root[:, 7:13] = torch.empty(len(env_ids), 6, device=env.device).uniform_(*env.cfg._fudan_task_cfg.env.reset_velocity_range)
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        action.reset_joint_positions[env_ids],
        torch.zeros_like(robot.data.default_joint_vel[env_ids]),
        env_ids=env_ids
    )

@requires_model_fields(
    'body_mass',
    'body_inertia',
    'body_ipos',
    'geom_friction',
    'geom_solref',
    recompute=RecomputeLevel.set_const
)
def randomize(env, env_ids):
    robot = env.scene['robot']
    model = env.sim.mj_model
    n = env.num_envs
    device = env.device
    domain = env.cfg._fudan_task_cfg.domain_rand
    ids = robot.indexing.body_ids
    base = int(ids[robot.body_names.index('base_link')])
    mass = env.sim.get_default_field('body_mass')[ids].expand(n, -1).clone()
    local = robot.body_names.index('base_link')
    add = torch.empty(n, device=device).uniform_(*domain.added_mass_range)
    mass[:, local] += add
    scale = torch.empty_like(mass).uniform_(*domain.mass_scale_range)
    com = torch.empty(n, 3, device=device).uniform_(*domain.com_range)
    buckets = torch.empty(domain.friction_buckets, device=device).uniform_(*domain.friction_range)
    friction = buckets[torch.randint(domain.friction_buckets, (n,), device=device)]
    restitution = torch.empty(n, device=device).uniform_(*domain.restitution_range)
    env.sim.model.body_mass[:, ids] = mass * scale
    env.sim.model.body_inertia[:, ids] = env.sim.get_default_field('body_inertia')[ids] * scale[..., None]
    env.sim.model.body_ipos[:, base] = env.sim.get_default_field('body_ipos')[base] + com
    geoms = torch.tensor(
        [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in set(ids.tolist())],
        device=device
    )
    env.sim.model.geom_friction[:, geoms, 0] = ((friction + 0.5) * 0.5)[:, None]
    effective = (restitution + 0.5) * 0.5
    log_e = effective.log()
    zeta = -log_e / (math.pi ** 2 + log_e.square()).sqrt()
    env.sim.model.geom_solref[:, geoms] = torch.stack((torch.full_like(zeta, -10000.0), -200 * zeta), -1)[:, None, :]
    env._fudan_parameters = dict(base_mass=mass[:, local], com=com, friction=friction, restitution=restitution)

def make_env_cfg(*, play=False, disturbances=None, task_cfg=None):
    task = FudanWalkCfg() if task_cfg is None else task_cfg
    cfg = make_huananhu_env_cfg(play=True)
    wheel_physics.record(task.sim.wheel_model)
    if task.sim.wheel_model=='mechanical':
        if (task.sim.wheel_motor_gain!=1. or task.sim.wheel_motor_delay_ms!=0. or
                task.sim.wheel_motor_time_constant_ms!=0. or task.sim.wheel_motor_torque_limit_nm is not None):
            raise ValueError('Mechanical wheels require unit output gain and no extra wheel response; use legacy_curve for historical replay')
        cfg.scene.entities['robot'].spec_fn=wheel_physics.mechanical_spec
    cfg.scene.num_envs = task.env.play_num_envs if play else task.env.num_envs
    cfg.sim.mujoco.timestep = task.sim.dt
    cfg.decimation = task.control.decimation
    cfg.episode_length_s = task.env.episode_length_s
    cfg._fudan_task_cfg = task
    cfg.commands = {}
    cfg.curriculum = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.events = {'reset_pose': EventTermCfg(func=reset_pose, mode='reset')}
    if not play:
        cfg.events['fudan_randomization'] = EventTermCfg(func=randomize, mode='startup')
    original = cfg.actions['pd']
    cfg.actions = {'pd': PhysicalActionCfg(
        entity_name='robot',
        nominal_leg_positions=task.control.nominal_leg_positions,
        task_cfg=task,
        leg_stiffness=task.control.leg_stiffness,
        leg_damping=task.control.leg_damping,
        leg_position_scale=task.control.leg_position_scale,
        wheel_velocity_scale=task.control.wheel_velocity_scale,
        max_wheel_velocity=task.control.max_wheel_velocity,
        wheel_stiffness=task.control.wheel_stiffness,
        wheel_damping=task.control.wheel_damping,
        motor_randomization=False,
        randomize_delays=False,
        action_delay_s=0.0,
        ground_start_hold_s=0.0,
        spring_force_bias_range=(0.0, 0.0),
        standup_landing_gate=False,
        standup_pushes=False,
        reference_randomization=not play,
        disturbances=disturbances or from_task_config(task),
        fitted_leg_load=task.sim.fitted_leg_load
    )}
    cfg.observations = {'actor': ObservationGroupCfg(
        terms={'sensors': ObservationTermCfg(func=current_sensors)},
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy='error'
    )}
    cfg.metrics = {'motor_sampling': MetricsTermCfg(func=sample_substep, per_substep=True, reduce='last')}
    pose = nominal_reset_pose()
    initial = cfg.scene.entities['robot'].init_state
    initial.pos = (0.0, 0.0, pose['height'])
    initial.joint_pos = {f'^{re.escape(name)}$': value for name, value in pose['joint_positions'].items()}
    return cfg

class FudanEnv:

    def __init__(
        self,
        num_envs=512,
        seed=1,
        device='cuda:0',
        randomized=True,
        physics_env=None,
        reset_contract=RESET_CONTRACT,
        disturbances=None,
        cfg=None
    ):
        if reset_contract not in (RESET_CONTRACT, RESET_V2_CONTRACT):
            raise ValueError('Unsupported PD-centred reset/history contract')
        self.cfg = FudanWalkCfg() if cfg is None else cfg
        self.posture_reward_profile = posture_reward.from_task_config(self.cfg)
        self.wheel_physics_profile = wheel_physics.record(self.cfg.sim.wheel_model)
        self.dt = policy_dt(self.cfg)
        self.max_episode_length = episode_steps(self.cfg)
        self.reset_contract = reset_contract
        from mjlab.envs import ManagerBasedRlEnv
        physics_cfg = make_env_cfg(play=not randomized, disturbances=disturbances, task_cfg=self.cfg)
        physics_cfg.scene.num_envs = num_envs
        physics_cfg.seed = seed
        self.env = ManagerBasedRlEnv(physics_cfg, device=device) if physics_env is None else physics_env
        num_envs, device = (self.env.num_envs, self.env.device)
        self.num_envs, self.device = (num_envs, device)
        self.robot = self.env.scene['robot']
        self.action = self.env.action_manager.get_term('pd')
        self.disturbance_config = self.action.cfg.disturbances
        if disturbances is not None and self.disturbance_config != disturbances:
            raise ValueError('Wrapped physics environment has a different disturbance configuration')
        self.update_push_scale()
        self.randomized = randomized
        self.commands = torch.zeros(num_envs, 3, device=device)
        self.history = torch.zeros(num_envs, 125, device=device)
        self._history_needs_fill = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.previous = torch.zeros(num_envs, 6, 2, device=device)
        self.previous_vel = torch.zeros(num_envs, 6, device=device)
        self.previous_position = torch.zeros(num_envs, 3, device=device)
        self.fail_ticks = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.sums = torch.zeros(num_envs, 2, device=device)
        self.speed_limit = self.cfg.commands.velocity_initial
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.total_steps = 0
        command_sampling.initialize(self)
        ids, _ = self.robot.find_joints(('lf0', 'lf1', 'rf0', 'rf1'), preserve_order=True)
        self.front_q = self.robot.indexing.joint_q_adr[ids]
        model = self.env.sim.mj_model
        joint_ids = [model.joint(f'robot/{name}').id for name in ('lf0', 'lf1', 'rf0', 'rf1')]
        limits = torch.tensor(model.jnt_range[joint_ids], device=device, dtype=torch.float32)
        mid = limits.mean(-1)
        half = (limits[:, 1] - limits[:, 0]) * self.cfg.rewards.soft_limit_half_span
        self.front_limits = torch.stack((mid - half, mid + half), -1)
        self.hip_ids = torch.tensor([model.joint('robot/' + n).id for n in ('lf0', 'rf0')], device=device)
        self.wheel_ids = torch.tensor([model.body('robot/' + n).id for n in ('lfwheel', 'rfwheel')], device=device)
        self.previous_position.copy_(self.robot.data.root_link_pos_w)
        self._update_state()

    def update_push_scale(self):
        if self.disturbance_config.pushes:
            mass = self.env.sim.model.body_mass[..., self.action._base_id].mean()
            self.action.push_force_scale.fill_(float(mass) * self.disturbance_config.max_push_vel / self.cfg.sim.dt)

    def resample(self, ids):
        ids = ids[(self.command_mode[ids] == 0) | (self.command_elapsed[ids] == 0)]
        self.commands[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(-self.speed_limit, self.speed_limit)
        self.commands[ids, 1] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.commands.yaw_range)
        self.commands[ids, 2] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.commands.height_range)
        command_sampling.apply_parking_commands(self, ids)

    def reset_idx(self, ids):
        if len(ids) == 0:
            return
        if (
            self.total_steps % self.max_episode_length == 0
            and (
                self.sums[ids, 0].mean() / self.max_episode_length > self.cfg.commands.curriculum_threshold * self.dt
                and self.sums[ids, 1].mean() / self.max_episode_length > self.cfg.commands.curriculum_threshold * self.dt * self.cfg.commands.curriculum_yaw_factor
            )
        ):
            self.speed_limit = min(
                self.cfg.commands.velocity_max,
                self.speed_limit + self.cfg.commands.velocity_increment
            )
        self.env.reset(env_ids=ids)
        command_sampling.reset_modes(self, ids)
        self.resample(ids)
        self.previous[ids] = 0.0
        self.previous_vel[ids] = 0.0
        self.fail_ticks[ids] = 0
        self.sums[ids] = 0.0
        self.episode_steps[ids] = 0
        # fresh reset先刷新速度参考并清动作；其他环境的历史不动。
        self.previous_position[ids] = self.robot.data.root_link_pos_w[ids]
        self._update_state(ids)
        if hasattr(self.action, 'disturbances'):
            self.action.disturbances.sensors[:, ids] = self.action.sensor_frame()[ids]
        if self.reset_contract == RESET_V2_CONTRACT:
            self.history[ids] = self.frame()[ids].repeat(1, 5)
        else:
            self._history_needs_fill[ids] = True

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # 只推进一次零动作10ms热步，之后由runner随机首轮进度。
        obs, critic, _, _, _, history = self.step(torch.zeros(self.num_envs, 6, device=self.device))
        return (obs, critic, history)

    def _update_state(self, ids=None):
        d = self.robot.data
        position = d.root_link_pos_w
        velocity = self.action.sample_velocity
        values = dict(
            position=position,
            linear=quat_apply_inverse(d.root_link_quat_w, (position - self.previous_position) / self.dt),
            angular=d.root_link_ang_vel_b,
            gravity=d.projected_gravity_b,
            height=position[:, 2] - self.env.scene.env_origins[:, 2],
            velocity=velocity,
            acceleration=(self.previous_vel - velocity) / self.dt
        )
        for name, value in values.items():
            if ids is None or not hasattr(self, name):
                setattr(self, name, value.clone())
            else:
                getattr(self, name)[ids] = value[ids]

    def frame(self):
        q = self.env.sim.data.qpos[:, self.action._motor_q_adr]
        return torch.cat(
            (
                self.angular * self.cfg.normalization.angular_velocity,
                self.gravity,
                self.commands * torch.tensor(self.cfg.normalization.command, device=self.device),
                (q - self.action.nominal_positions)[:, LEGS],
                self.velocity * self.cfg.normalization.joint_velocity,
                self.action.raw_action
            ),
            -1
        )

    def observations(self):
        clean = self.frame()
        p = getattr(self.env, '_fudan_parameters', None)
        if p is None:
            mass = torch.zeros(self.num_envs, 1, device=self.device)
            com = torch.zeros(self.num_envs, 3, device=self.device)
            friction = torch.full_like(mass, 0.5)
            restitution = torch.full_like(mass, 0.5)
        else:
            mass = (p['base_mass'] - p['base_mass'].mean())[:, None]
            com = p['com']
            friction = p['friction'][:, None]
            restitution = p['restitution'][:, None]
        critic = torch.cat(
            (
                self.linear * self.cfg.normalization.command[0],
                clean,
                self.previous[:, :, 0],
                self.previous[:, :, 1],
                self.acceleration * self.cfg.normalization.acceleration,
                ((self.height - self.cfg.normalization.height_center).clamp(-1.0, 1.0) * self.cfg.normalization.height_scale)[:, None].expand(-1, 77),
                self.action.torques * self.cfg.normalization.torques,
                mass,
                com,
                self.action.nominal_offset,
                friction,
                restitution
            ),
            -1
        ).clamp(-self.cfg.normalization.clip, self.cfg.normalization.clip)
        obs = clean.clone()
        if getattr(self, 'disturbance_config', DisturbanceConfig()).sensor_delay:
            obs[:, SENSOR_SLOTS] = self.action.disturbances.observed()
        if self.randomized:
            scales = obs.new_tensor(self.cfg.normalization.noise_scales)
            noise = (torch.rand_like(obs) * 2 - 1) * scales
            if hasattr(self, 'noise_mask'):
                noise *= self.noise_mask[:, None]
            obs += noise
        clipped = obs.clamp(-self.cfg.normalization.clip, self.cfg.normalization.clip)
        self._history_frame = obs if self.reset_contract == RESET_V2_CONTRACT else clipped
        return (clipped, critic)

    def reward(self):
        error = (self.commands[:, 0] - self.linear[:, 0]).square()
        yaw_error = (self.commands[:, 1] - self.angular[:, 2]).square()
        delta = self.env.sim.data.xpos[:, self.wheel_ids] - self.env.sim.data.xanchor[:, self.hip_ids]
        quaternion = self.robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        local = quat_apply_inverse(quaternion.reshape(-1, 4), delta.reshape(-1, 3)).reshape(-1, 2, 3)
        theta = torch.atan2(local[:, :, 0], -local[:, :, 2])
        q = self.env.sim.data.qpos[:, self.front_q]
        limit = ((self.front_limits[:, 0] - q).clamp_min(0.0) + (q - self.front_limits[:, 1]).clamp_min(0.0)).sum(-1)
        action = self.action.raw_action
        raw = dict(
            tracking_lin_vel=torch.exp(-error / self.cfg.rewards.tracking_sigma),
            tracking_lin_vel_enhance=torch.exp(-error / self.cfg.rewards.tracking_enhance_sigma) - 1.0,
            tracking_ang_vel=torch.exp(-yaw_error / self.cfg.rewards.tracking_sigma),
            tracking_ang_vel_enhance=torch.exp(-yaw_error / self.cfg.rewards.tracking_enhance_sigma) - 1.0,
            base_height=torch.exp(-(self.height - self.commands[:, 2]).square() / self.cfg.rewards.height_sigma),
            nominal_state=(theta[:, 0] - theta[:, 1]).square(),
            lin_vel_z=self.linear[:, 2].square(),
            ang_vel_xy=self.angular[:, :2].square().sum(-1),
            **posture_reward.penalties(self),
            dof_vel=self.velocity[:, LEGS].square().sum(-1),
            dof_acc=self.acceleration.square().sum(-1),
            torques=self.action.torques.square().sum(-1),
            action_rate=(action - self.previous[:, :, 0]).square().sum(-1),
            action_smooth=(action[:, LEGS] - 2 * self.previous[:, LEGS, 0] + self.previous[:, LEGS, 1]).square().sum(-1),
            collision=torch.zeros_like(error),
            dof_pos_limits=limit
        )
        terms = {k: (v * self.cfg.rewards.scales[k] * self.dt).clamp(-self.dt, self.dt) for k, v in raw.items()}
        self.sums[:, 0] += terms['tracking_lin_vel']
        self.sums[:, 1] += terms['tracking_ang_vel']
        return (torch.stack(list(terms.values())).sum(0), terms)

    def step(self, actions, *, auto_reset=True, resample=True):
        if not torch.isfinite(actions).all():
            raise FloatingPointError('Non-finite policy action')
        self.env.step(actions)
        self.total_steps += 1
        self.episode_steps += 1
        self.command_elapsed += 1
        self._update_state()
        if not torch.isfinite(self.env.sim.data.qpos).all() or not torch.isfinite(self.env.sim.data.qvel).all():
            raise FloatingPointError('Non-finite physical state')
        if resample:
            self.resample((self.episode_steps.remainder(self.cfg.commands.resampling_steps) == 0).nonzero().flatten())
            command_sampling.update_stops(self)
        bad = self.gravity[:, 2] > self.cfg.env.terminal_gravity_z
        self.fail_ticks = torch.where(bad, self.fail_ticks + 1, 0)
        timeout = self.episode_steps > self.max_episode_length
        done = (self.fail_ticks > self.cfg.env.fail_ticks) | timeout
        reward, terms = self.reward()
        if auto_reset and done.any():
            self.reset_idx(done.nonzero().flatten())
        obs, critic = self.observations()
        # 原FIFO只推进一帧；v2保存裁剪前帧，v3保存当前带噪裁剪帧。
        self.history = torch.cat((self.history[:, 25:], self._history_frame), -1)
        if self.reset_contract == RESET_CONTRACT:
            self.history[self._history_needs_fill] = self._history_frame[self._history_needs_fill].repeat(1, 5)
            self._history_needs_fill.zero_()
        self.previous[:, :, 1] = self.previous[:, :, 0]
        self.previous[:, :, 0] = self.action.raw_action
        self.previous_vel.copy_(self.velocity)
        self.previous_position.copy_(self.position)
        return (obs, critic, reward, done, {'time_outs': timeout, 'terms': terms}, self.history)

    def close(self):
        self.env.close()
