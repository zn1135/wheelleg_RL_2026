"""单级平台环境，复用平地的实体控制/历史；地形真值只进入Critic。"""
import math
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.spec_config import GeomCfg
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv, make_env_cfg, randomize, reset_pose
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT
from .config import TASK, FudanStepCfg
from .terrain import PlatformCfg, ElevatedGrid
from .contacts import ContactMonitor, sample_contacts
from .reference_motion import MotionHint


@requires_model_fields('body_mass', 'body_inertia', 'body_ipos', 'geom_friction',
                       'geom_solref', recompute=RecomputeLevel.set_const)
def randomize_step(env, env_ids):
    model = env.sim.mj_model
    terrain_body = model.body('terrain').id
    env._terrain_friction_ids = torch.tensor([g for g in range(model.ngeom)
        if model.geom_bodyid[g] == terrain_body and (model.geom_contype[g] or model.geom_conaffinity[g])],
        device=env.device)
    if not len(env._terrain_friction_ids):
        raise ValueError('Generated platform has no collision geometry')
    if env.cfg._step_randomized:
        randomize(env, env_ids)
    else:
        # 名义评估仍展开材料字段，允许逐样本固定真实接触摩擦。
        robot = env.scene['robot']
        bodies = robot.indexing.body_ids
        base = int(bodies[robot.body_names.index('base_link')])
        coefficient = torch.ones(env.num_envs, device=env.device)
        geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in set(bodies.tolist())]
        env.sim.model.geom_friction[:, geoms, 0] = coefficient[:, None]
        env.sim.model.geom_friction[:, env._terrain_friction_ids, 0] = coefficient[:, None]
        env._fudan_parameters = dict(base_mass=env.sim.model.body_mass[:, base].expand(env.num_envs).clone(),
            com=torch.zeros(env.num_envs, 3, device=env.device), friction=coefficient,
            restitution=torch.full_like(coefficient, .5))


def reset_step_pose(env, env_ids):
    reset_pose(env, env_ids)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    types = env.scene.terrain.terrain_types[env_ids]
    levels = env.scene.terrain.terrain_levels[env_ids]
    max_level = max(1, len(env.cfg._fudan_task_cfg.step.heights)-1)
    angle = env.cfg._fudan_task_cfg.step.approach_angle_deg * levels/max_level
    delta = torch.deg2rad(angle) * (2*torch.rand(len(env_ids), device=env.device)-1)
    if hasattr(env, '_step_fixed_yaw'):
        delta = env._step_fixed_yaw[env_ids]
    reverse = env._step_reverse[env_ids] if hasattr(env, '_step_reverse') else torch.zeros_like(types, dtype=torch.bool)
    yaw = torch.where(types == 2, math.pi, 0.) + reverse*math.pi + delta
    root = env.scene['robot'].data.default_root_state[env_ids].clone()
    root[:, :3] = env.scene.env_origins[env_ids]
    # 斜向接近的直线在台阶中间相交；下台出生距离缩短，保证双轮仍在台面上。
    direction = torch.where(types == 2, -1., 1.)
    distance = root.new_tensor(env.cfg._fudan_task_cfg.step.approach_distance)[types]
    root[:, 1] -= direction*distance*torch.tan(delta)
    root[:, 2] += env.action_manager.get_term('pd').reset_root_height[env_ids]
    root[:, 3:7] = 0.
    root[:, 3], root[:, 6] = torch.cos(yaw/2), torch.sin(yaw/2)
    root[:, 7:] = 0.
    env.scene['robot'].write_root_state_to_sim(root, env_ids=env_ids)


def make_step_env_cfg(cfg, *, randomized, num_envs, seed):
    physics = make_env_cfg(play=not randomized, disturbances=from_task_config(cfg), task_cfg=cfg)
    physics.scene.num_envs, physics.seed = num_envs, seed
    physics._step_randomized = randomized
    step = cfg.step
    physics.sim.nconmax = step.contact_capacity
    physics.sim.njmax = step.constraint_capacity
    terrain = physics.scene.terrain
    terrain.terrain_type = 'generator'
    terrain.max_init_terrain_level = step.start_level
    terrain.materials = ()
    terrain.textures = ()
    terrain.geoms = (GeomCfg(geom_names_expr=('.*',), friction=(1., .005, .0001)),)
    terrain.terrain_generator = TerrainGeneratorCfg(seed=seed, curriculum=True,
        border_width=step.border_width,
        size=step.patch_size, num_rows=len(step.heights), num_cols=3, color_scheme='none',
        sub_terrains={mode: PlatformCfg(mode=mode, proportion=weight, heights=step.heights,
            width=step.width, depth=step.depth, side_obstacle_level=step.side_obstacle_level,
            side_obstacle_height=step.side_obstacle_height, approach_distance=step.approach_distance)
            for mode, weight in zip(('flat', 'up', 'down'), step.terrain_weights)})
    physics.events['reset_pose'] = EventTermCfg(func=reset_step_pose, mode='reset')
    physics.events['fudan_randomization'] = EventTermCfg(func=randomize_step, mode='startup')
    sensors = []
    for name, pattern, slots in (
        ('step_base', '^base_link$', 1),
        ('step_legs', '^[lr]f(0|1|00|01|02|03)$', 1),
        ('step_wheels', '^(lfwheel|rfwheel)$', 4),
    ):
        sensors.append(ContactSensorCfg(name=name,
            primary=ContactMatch(mode='body', pattern=pattern, entity='robot'),
            secondary=ContactMatch(mode='body', pattern='terrain'),
            fields=('found', 'force', 'pos', 'normal', 'tangent'),
            reduce='maxforce', num_slots=slots, global_frame=True))
    sensors.append(RayCastSensorCfg(name='step_scan',
        frame=ObjRef(type='body', name='base_link', entity='robot'),
        pattern=ElevatedGrid(size=(1., .6), resolution=.1), ray_alignment='yaw',
        max_distance=3., include_geom_groups=(0,)))
    sensors.append(RayCastSensorCfg(name='step_support', frame=tuple(
        ObjRef(type='body', name=n, entity='robot') for n in ('lfwheel', 'rfwheel')),
        pattern=ElevatedGrid(size=(0., 0.), resolution=.1), ray_alignment='world',
        max_distance=3., include_geom_groups=(0,)))
    physics.scene.sensors = tuple(sensors)
    physics.metrics['platform_contacts'] = MetricsTermCfg(func=sample_contacts, per_substep=True, reduce='max')
    return physics


class StepEnv(FudanEnv):
    task_name = TASK

    def __init__(self, num_envs=4096, seed=1, device='cuda:0', randomized=True,
                 cfg=None, reset_contract=RESET_CONTRACT, **kwargs):
        cfg = FudanStepCfg() if cfg is None else cfg
        physics = ManagerBasedRlEnv(make_step_env_cfg(cfg, randomized=randomized,
                                    num_envs=num_envs, seed=seed), device=device)
        try:
            super().__init__(num_envs=num_envs, seed=seed, device=device, randomized=randomized,
                cfg=cfg, reset_contract=reset_contract, physics_env=physics, **kwargs)
        except BaseException:
            physics.close()
            raise
        self.contacts = ContactMonitor(physics, cfg.step)
        physics._step_contacts = self.contacts
        self.reverse = torch.zeros(num_envs, device=device, dtype=torch.bool)
        physics._step_reverse = self.reverse
        self.terrain = physics.scene.terrain
        # 固定20/40/40环境类型，避免下台的课程等级串到上台。
        types = (torch.arange(num_envs, device=device)*10//num_envs)
        self.terrain.terrain_types[:] = torch.where(types < 2, 0, torch.where(types < 6, 1, 2))
        self.terrain.terrain_levels.fill_(cfg.step.start_level)
        self._set_origins(torch.arange(num_envs, device=device))
        self.parking_time = torch.zeros(num_envs, device=device)
        self.parking_ticks = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.landing_ticks = torch.zeros_like(self.parking_ticks)
        self.ray_failed = torch.zeros(num_envs,device=device,dtype=torch.bool)
        self.max_approach_speed = torch.zeros_like(self.parking_time)
        self.crossed = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.success = torch.zeros_like(self.crossed)
        self.finished = torch.zeros_like(self.crossed)
        self.episode_rows = []
        self.motion_hint=(MotionHint(cfg.step,device) if cfg.rewards.scales.get('reference_motion',0.) else None)
        self.command_sampling.update(version='fudan-platform-commands-v4-speed-cap',
            terrain_probabilities=list(cfg.step.terrain_weights),
            terrain_assignment=cfg.step.terrain_assignment,
            command_curriculum=cfg.step.command_curriculum,
            approach_speed_min=cfg.step.approach_speed[0],
            speed_cap_initial=cfg.commands.velocity_initial,
            speed_cap_max=cfg.commands.velocity_max,
            speed_cap_increment=cfg.commands.velocity_increment,
            reverse_probability=cfg.step.reverse_probability,
            platform_height_range=list(cfg.step.height_range),
            height_sampling='uniform_independent_of_terrain_level',
            flat_behavior_probabilities=dict(drive=.5,stand=.25,drive_then_stop=.25),
            legacy_mode_sampling=False)
        self._step_ready = True
        self._update_state()

    def _set_origins(self, ids):
        self.terrain.env_origins[ids] = self.terrain.terrain_origins[
            self.terrain.terrain_levels[ids], self.terrain.terrain_types[ids]]

    @property
    def modes(self):
        return self.terrain.terrain_types

    def _temporary_support_allowed(self):
        return (self.modes!=0)&getattr(self.cfg.step,'allow_temporary_support',False)

    def _contact_failure(self):
        return self.contacts.failed&~self._temporary_support_allowed()

    def reset_idx(self, ids):
        if not len(ids):
            return
        if self.cfg.step.curriculum:
            scored = self.finished[ids] & (self.modes[ids] != 0)
            # 参考复旦按通行进度升级；最终成功仍独立要求停车2秒。
            promote = self.crossed[ids] & ~self._contact_failure()[ids] & (
                self.max_approach_speed[ids]<=self.cfg.step.max_approach_speed) & ~self.ray_failed[ids]
            delta = torch.where(promote, 1, -1)*scored
            levels=self.terrain.terrain_levels[ids]+delta
            maximum=len(self.cfg.step.heights)
            solved_last=scored & promote & (levels>=maximum)
            levels=torch.where(solved_last,torch.randint(self.cfg.step.minimum_stair_level,maximum,
                (len(ids),),device=self.device),levels)
            # 台阶类型不能退成零高度；平地回练由独立20%环境承担。
            minimum=torch.where(self.modes[ids]!=0,self.cfg.step.minimum_stair_level,0)
            self.terrain.terrain_levels[ids] = torch.maximum(levels.clamp(0,maximum-1),minimum)
        self.reverse[ids] = (torch.rand(len(ids), device=self.device)<self.cfg.step.reverse_probability
                             if self.randomized else False)
        self.reverse[ids] &= self.modes[ids]!=0
        if hasattr(self.env, '_step_fixed_reverse'):
            self.reverse[ids] = self.env._step_fixed_reverse[ids] & (self.modes[ids]!=0)
        self._set_origins(ids)
        self.contacts.reset(ids)
        self.parking_time[ids] = 0.
        self.parking_ticks[ids] = 0
        self.landing_ticks[ids] = 0
        self.ray_failed[ids] = False
        self.max_approach_speed[ids] = 0.
        self.crossed[ids] = False
        self.success[ids] = False
        self.finished[ids] = False
        super().reset_idx(ids)

    def resample(self, ids):
        if not hasattr(self, '_step_ready'):
            return super().resample(ids)
        self.command_elapsed[ids] = 0
        # 按已学会的速度上限采样，不能一开始全部给最大速度。
        low, high = self.cfg.step.approach_speed
        high = min(high, self.speed_limit)
        self.commands[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(low, high)
        self.commands[ids, 0] *= torch.where(self.reverse[ids], -1., 1.)
        self.commands[ids, 1] = 0.
        # 独立抽取高度目标，不通过命令槽向盲策略暗示真实台阶高度。
        self.commands[ids, 2] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.step.height_range)
        self.command_mode[ids] = 0
        flat = ids[self.modes[ids] == 0]
        if len(flat):
            # 台阶提速不改变平地保持回合原有的低速分布。
            self.commands[flat, 0] = torch.empty(len(flat), device=self.device).uniform_(*self.cfg.commands.stop_speed_range)
            self.commands[flat, 0] *= torch.where(torch.rand(len(flat), device=self.device)<.5, -1., 1.)
            self.commands[flat, 1] = torch.empty(len(flat), device=self.device).uniform_(*self.cfg.commands.yaw_range)
            self.commands[flat, 2] = torch.empty(len(flat), device=self.device).uniform_(*self.cfg.commands.height_range)
            # 平地一半普通行驶，四分之一站立，四分之一行驶后停车。
            draw = torch.rand(len(flat), device=self.device)
            stand, stop = flat[draw<.25], flat[(draw>=.25)&(draw<.5)]
            self.command_mode[stand], self.command_mode[stop] = 1, 2
            self.commands[stand, :2] = 0.
            self.stop_command_step[stop] = torch.randint(300, 501, (len(stop),), device=self.device)
        if hasattr(self, 'fixed_commands'):
            self.commands[ids] = self.fixed_commands[ids]

    def _update_state(self, ids=None):
        super()._update_state(ids)
        if not hasattr(self, '_step_ready'):
            return
        scan = self.env.scene['step_scan'].data
        support = self.env.scene['step_support'].data
        missed = (scan.distances<0).any(-1)|(support.distances<0).any(-1)
        if not hasattr(self,'terrain_samples'):
            if missed.any():raise FloatingPointError('Initial platform rays missed terrain')
            self.terrain_samples=scan.hit_pos_w[...,2].clone()
            self.support_height=support.hit_pos_w[...,2].clone()
        # 缺测世界明确失败；旧值只维持终止拍张量有限，不当作真实平地或成功证据。
        self.ray_failed |= missed
        self.terrain_samples = torch.where(missed[:,None],self.terrain_samples,scan.hit_pos_w[...,2])
        self.support_height = torch.where(missed[:,None],self.support_height,support.hit_pos_w[...,2])
        reference = self.position[:, 2]-self.support_height.mean(-1)
        if ids is None:
            self.height.copy_(reference)
        else:
            self.height[ids] = reference[ids]
        self.normal_support = (self.contacts.supported.all(-1) & ~self.contacts.side_contact &
                               ~self.contacts.step_contact &
                               ((self.support_height[:, 0]-self.support_height[:, 1]).abs()<.02) & ~self.ray_failed)

    def observations(self):
        obs, critic = super().observations()
        # 同样是77槽；平地时逐值等价，Actor仍只读取原25维本体反馈。
        heights = ((self.position[:, 2, None]-self.terrain_samples-self.cfg.normalization.height_center)
                   .clamp(-1., 1.)*self.cfg.normalization.height_scale)
        # 3速度+25当前帧+两帧各6动作+6加速度=46；随后77槽，不能覆盖力矩。
        critic[:, 46:123] = heights
        return obs, critic

    def reward(self):
        _, terms = super().reward()
        direction = torch.where(self.modes == 2, -1., 1.)
        edge = self.env.scene.env_origins[:, 0] + direction*self.position.new_tensor(self.cfg.step.approach_distance)[self.modes]
        near_edge = (self.modes != 0) & ((self.position[:, 0]-edge).abs()<.55)
        # 南科大原版高度指数项，保留局部地面参考和逐项±dt截断。
        error=(self.height-self.commands[:,2]).square()
        terms['base_height']=(torch.exp(-error/self.cfg.rewards.height_sigma)*
            self.cfg.rewards.scales['base_height']*self.dt).clamp(-self.dt,self.dt)
        terms['base_height_enhance']=((torch.exp(-error/(10*self.cfg.rewards.height_sigma))-1)*
            self.cfg.rewards.scales['base_height_enhance']*self.dt).clamp(-self.dt,self.dt)
        # 触沿前即可差动抬轮；不指定左/右先行，也不要求两轮同步。
        terms['nominal_state'] *= torch.where(near_edge, self.cfg.step.crossing_symmetry_factor, 1.)
        terms['lin_vel_z'] *= ~near_edge
        allowed=self._temporary_support_allowed()
        terms['collision'] = -self.dt*self.contacts.step_contact.float()*~allowed
        if getattr(self.cfg.step,'allow_temporary_support',False):
            traversing=allowed&~self.crossed
            # 越障阶段不要求保持高站姿或左右对称；保留横滚约束，俯仰自由探索。
            for name in ('base_height','base_height_enhance','nominal_state','lin_vel_z'):
                terms[name]=terms[name].masked_fill(traversing,0.)
            roll_only=(self.gravity[:,1].square()*self.cfg.rewards.scales['orientation']*self.dt).clamp(-self.dt,self.dt)
            terms['orientation']=torch.where(traversing,roll_only,terms['orientation'])
            terms['reference_motion']=torch.zeros_like(self.height)
            if self.motion_hint is not None:
                legs=self.action._leg_action_ids
                position=self.env.sim.data.qpos[:,self.action._motor_q_adr[legs]]
                velocity=self.env.sim.data.qvel[:,self.action._motor_v_adr[legs]]
                terms['reference_motion']=(self.motion_hint.score(position,velocity)*near_edge*traversing*
                    self.cfg.rewards.scales['reference_motion']*self.dt)
            target_h=torch.where(self.modes==2,0.,self.position.new_tensor(self.cfg.step.heights)[self.terrain.terrain_levels])
            dist=(self.env.sim.data.xpos[:,self.wheel_ids,0]-edge[:,None])*direction[:,None]
            loaded=((dist>.06)&((self.support_height-target_h[:,None]).abs()<.01)&self.contacts.supported)
            terms['target_support']=loaded.float().mean(-1)*allowed*self.cfg.rewards.scales['target_support']*self.dt
        terms = {key:value.masked_fill(self.ray_failed,0.) for key,value in terms.items()}
        return torch.stack(list(terms.values())).sum(0), terms

    def _progress(self):
        origin = self.env.scene.env_origins
        direction = torch.where(self.modes == 2, -1., 1.)
        edge = origin[:, 0] + direction*origin.new_tensor(self.cfg.step.approach_distance)[self.modes]
        wheels = self.env.sim.data.xpos[:, self.wheel_ids]
        dist = (wheels[..., 0]-edge[:, None])*direction[:, None]
        target_h = torch.where(self.modes == 2, 0., self.parking_time.new_tensor(self.cfg.step.heights)[self.terrain.terrain_levels])
        on_target = ((self.support_height-target_h[:, None]).abs()<.01).all(-1)
        inside_lane = ((wheels[..., 1]-origin[:, None, 1]).abs()<self.cfg.step.width/2-.04).all(-1)
        reached = (dist.amin(-1)>getattr(self.cfg.step,'landing_clearance_m',.45)) & on_target & inside_lane & self.contacts.supported.all(-1)
        self.landing_ticks = torch.where(reached & (self.modes!=0), self.landing_ticks+1, 0)
        self.crossed |= self.landing_ticks>=math.ceil(self.cfg.step.landing_s/self.dt)
        speed = self.robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
        approach = (dist.amin(-1)>-.5)&(dist.amin(-1)<.1)&(self.modes!=0)
        self.max_approach_speed = torch.maximum(self.max_approach_speed,
            torch.where(approach, self.contacts.step_speed_peak, 0.))
        roll = torch.atan2(-self.gravity[:, 1], -self.gravity[:, 2]).abs()
        pitch = torch.asin(self.gravity[:, 0].clamp(-1., 1.)).abs()
        stable = (self.crossed & (self.commands[:, :2].abs().amax(-1)<1e-6) &
                  self.normal_support & ~self.contacts.step_contact &
                  (speed<=self.cfg.step.parking_speed) & (roll<=math.radians(self.cfg.step.parking_angle_deg)) &
                  (pitch<=math.radians(self.cfg.step.parking_angle_deg)))
        self.parking_ticks = torch.where(stable, self.parking_ticks+1, 0)
        self.parking_time = self.parking_ticks*self.dt
        self.success = (self.parking_ticks>=math.ceil(self.cfg.step.parking_s/self.dt))&(
            self.max_approach_speed<=self.cfg.step.max_approach_speed)&~self._contact_failure()
        relative = self.position[:, :2]-origin[:, :2]
        # 边界失败不能让机器人绕过平台或落出训练块仍获得成功。
        outside = (relative[:, 1].abs()>1.6) | (relative[:, 0].abs()>3.5)
        missed_lane = (self.modes!=0)&(dist.amax(-1)>-.05)&~inside_lane
        too_far = (self.modes==1)&(dist.amax(-1)>self.cfg.step.depth-.25)
        return outside|missed_lane|too_far

    def step(self, actions, *, auto_reset=True, resample=True):
        self.contacts.step_contact.zero_()
        self.contacts.step_speed_peak.zero_()
        self.contacts.step_forces.clear()
        self.contacts.step_positions.clear()
        # 复用原物理/历史更新，延后复位以便先保存台阶失败及接触证据。
        obs, critic, reward, done, info, history = super().step(actions, auto_reset=False, resample=False)
        if self.contacts.capacity_overflow.any():
            raise RuntimeError('Step simulation contact/constraint capacity overflow; cannot train with truncated contacts')
        outside = self._progress()
        contact_failure=self._contact_failure()
        failure = contact_failure | outside | (self.max_approach_speed>self.cfg.step.max_approach_speed) | self.ray_failed
        done |= failure | self.success
        info['time_outs'] &= ~failure & ~self.success
        if resample:
            flat_resample = ((self.modes==0)&(self.command_mode==0)&
                             (self.command_elapsed>=self.cfg.commands.resampling_steps))
            self.resample(flat_resample.nonzero().flatten())
            stop = ((self.command_mode==2)&(self.command_elapsed>=self.stop_command_step)) | self.crossed
            self.commands[stop, :2] = 0.
        success = self.success.clone()
        terrain_valid = ~self.ray_failed.clone()
        ids = done.nonzero().flatten()
        new_ids = (done & ~self.finished).nonzero().flatten()
        if len(new_ids):
            self.finished[new_ids] = True
            metrics = self.contacts.metrics(new_ids)
            for j, idx in enumerate(new_ids.tolist()):
                row=dict(env=idx, mode=int(self.modes[idx]),
                    level=int(self.terrain.terrain_levels[idx]), success=bool(self.success[idx]),
                    reverse=bool(self.reverse[idx]), crossed=bool(self.crossed[idx]),
                    strict_contact_failure=bool(self.contacts.failed[idx]),
                    contact_failure_applied=bool(contact_failure[idx]),
                    ray_missed=bool(self.ray_failed[idx]),
                    failure=bool(failure[idx]), max_approach_speed=float(self.max_approach_speed[idx]),
                    **{key: value[j].cpu().tolist() for key, value in metrics.items()})
                if self.ray_failed[idx]:
                    scan=self.env.scene['step_scan'];support=self.env.scene['step_support']
                    row['ray_failure_state']=dict(qpos=self.env.sim.data.qpos[idx].cpu().tolist(),
                        relative_root=(self.position[idx]-self.env.scene.env_origins[idx]).cpu().tolist(),
                        gravity=self.gravity[idx].cpu().tolist(),
                        scan_misses=int((scan.data.distances[idx]<0).sum()),
                        support_misses=int((support.data.distances[idx]<0).sum()),
                        scan_origins_min=scan._cached_world_origins[idx].amin(0).cpu().tolist(),
                        scan_origins_max=scan._cached_world_origins[idx].amax(0).cpu().tolist())
                self.episode_rows.append(row)
        if auto_reset and len(ids):
            self.reset_idx(ids)
        # 复位后替换最后一帧；没有复位时只改无噪声的命令槽。
        if auto_reset and len(ids):
            obs, critic = self.observations()
            self.history[:, -25:] = self._history_frame
            if auto_reset and len(ids):
                self.history[ids] = self._history_frame[ids].repeat(1, 5)
                self._history_needs_fill[ids] = False
        elif resample:
            command = self.commands*self.commands.new_tensor(self.cfg.normalization.command)
            obs[:, 6:9] = command
            critic[:, 9:12] = command
            self.history[:, -19:-16] = command
        info['step_success'] = success
        info['terrain_valid'] = terrain_valid
        info['step_failure'] = failure.clone()
        return obs, critic, reward, done, info, self.history
