"""Flat-ground Jump policy and physical instrumentation for the obstacle replay."""
import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv, make_env_cfg
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.landing import Landing, install_landing_metrics
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import (BAR_DEPTH, BAR_HALF_WIDTH, BAR_X, EXTEND, FLIGHT_HEIGHT,
                     HEIGHTS, TASK, TUCK, jump_config)


def sensor(name, bodies, secondary, *, found_only=False):
    return ContactSensorCfg(name=name, primary=ContactMatch(mode='body', entity='robot', pattern=bodies),
        secondary=ContactMatch(mode='geom', pattern=secondary),
        fields=('found',) if found_only else ('force',), reduce='maxforce' if found_only else 'netforce')


def capture_substep(physics):
    # 取每次物理积分后的接触，避免漏掉一次策略步内短暂碰条或机身触地。
    env = getattr(physics, '_jump_trace_env', None)
    if env is not None:
        sample = env.trace_frame()
        env.substeps.append(sample)
        return sample[:, 20]
    return physics.sim.data.qpos[:, 0] * 0.


class JumpEnv(FudanEnv):
    task_name = TASK

    def __init__(self, num_envs=20, seed=1, device='cuda:0', randomized=True,
                 cfg=None, bar_height=None, record_flat=False, **kwargs):
        cfg = jump_config() if cfg is None else cfg
        physical = make_env_cfg(play=not randomized, task_cfg=cfg)
        physical.scene.num_envs, physical.seed = num_envs, seed
        sensors = [sensor('jump_wheel', '^(lfwheel|rfwheel)$', 'terrain'),
                   sensor('jump_ground', '^(base_link|[lr]f(0|1|00|01|02|03))$', 'terrain')]
        if bar_height is not None:
            if bar_height not in HEIGHTS:
                raise ValueError('Unsupported bar height')
            def add_bar(spec):
                spec.body('terrain').add_geom(name='jump_bar', type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(BAR_DEPTH/2, BAR_HALF_WIDTH, bar_height/2),
                    pos=(BAR_X + BAR_DEPTH/2, 0., bar_height/2), rgba=(.9,.32,.08,1.))
            physical.scene.spec_fn = add_bar
            sensors.append(sensor('jump_bar_touch', '.*', 'jump_bar', found_only=True))
        self.has_bar = bar_height is not None
        if self.has_bar or record_flat:
            physical.scene.env_spacing = 0.
            physical.scene.terrain.env_spacing = 0.
            physical.metrics['jump_trace'] = MetricsTermCfg(func=capture_substep, per_substep=True, reduce='last')
        landing_cfg = getattr(cfg.rewards,'landing',None)
        if landing_cfg:
            install_landing_metrics(physical)
        physical.scene.sensors = tuple(sensors)
        physics = ManagerBasedRlEnv(physical, device=device)
        try:
            super().__init__(num_envs, seed, device, randomized, physics_env=physics, cfg=cfg, **kwargs)
            self.wheel_meshes = [self._collision_mesh(body) for body in ('lfwheel','rfwheel')]
        except BaseException:
            physics.close()
            raise
        self.last_contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.air_integral = self.height.new_zeros(num_envs)
        self.grounded = self.height.new_zeros(num_envs, dtype=torch.bool)
        self.in_jump = self.grounded.clone()
        self.landing_ticks = self.episode_steps.clone()
        self.flight_ticks = self.episode_steps.clone()
        self.peak_clearance = self.height.new_zeros(num_envs)
        self.totals = self.height.new_zeros(5, dtype=torch.float64)
        self.landing = Landing(self,landing_cfg) if landing_cfg else None
        self.train_peak = 'wheel_peak_gain' in cfg.rewards.scales
        if self.train_peak:
            self.highest_landed_peak = self.height.new_zeros(())
        self.substeps = []
        if self.has_bar or record_flat:
            physics._jump_trace_env = self

    def _collision_mesh(self, body):
        model = self.env.sim.mj_model
        bid = model.body('robot/' + body).id
        (gid,) = np.flatnonzero((model.geom_bodyid == bid) &
                                ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        mid = model.geom_dataid[gid]
        start = model.mesh_vertadr[mid]
        pts = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
        # 垂直投影最低点必在凸包上。
        verts = torch.tensor(pts[ConvexHull(pts).vertices].copy(),device=self.device).T.contiguous()
        return gid,verts

    def mesh_bottom(self, mesh):
        gid,verts = mesh
        sim = self.env.sim.data
        rotation_z = sim.geom_xmat[:,gid].reshape(-1,3,3)[:,2]
        return sim.geom_xpos[:,gid,2] + (rotation_z @ verts).amin(-1)

    def wheel_bottom(self):
        return torch.stack([self.mesh_bottom(mesh) for mesh in self.wheel_meshes],-1)

    def ground_contacts(self):
        wheels = self.env.scene['jump_wheel'].data.force[:, :, 2].abs() > 1.
        other = self.env.scene['jump_ground'].data.force.norm(dim=-1)
        return wheels, other

    def trace_frame(self):
        wheels, other = self.ground_contacts()
        sim = self.env.sim.data
        bar_touch = ((self.env.scene['jump_bar_touch'].data.found > 0).any(1, keepdim=True)
                     if self.has_bar else torch.zeros_like(wheels[:, :1]))
        return torch.cat((sim.xpos[:, self.wheel_ids].flatten(1), self.wheel_bottom(),
            wheels.float(), other.amax(-1, keepdim=True), self.robot.data.projected_gravity_b,
            self.robot.data.root_link_pos_w, self.robot.data.root_link_lin_vel_w,
            bar_touch,
            self.robot.data.root_link_lin_vel_b[:, :1], self.robot.data.root_link_ang_vel_b[:, 2:3]), 1)

    def reset_idx(self, ids):
        super().reset_idx(ids)
        for item in (self.last_contact, self.air_integral, self.grounded, self.in_jump,
                     self.landing_ticks, self.flight_ticks, self.peak_clearance):
            item[ids] = 0
        if self.landing is not None:
            self.landing.reset(ids)

    def track_jumps(self, contact, flight, vz, other, bottom):
        # 先双轮承重，再向上脱离地面；出生下落不能产生起跳事件。
        clean = (other.amax(-1) <= .1) & (self.gravity[:, 2] < -.9)
        takeoff = self.grounded & flight & (vz > 0) & clean & ~self.in_jump
        self.grounded = torch.where(flight | ~clean, False, self.grounded | contact.all(-1))
        self.in_jump |= takeoff
        self.flight_ticks += (self.in_jump & flight).long()
        clearance = bottom.amin(-1) - self.env.scene.env_origins[:, 2]
        self.peak_clearance = torch.where(self.in_jump & flight,
            torch.maximum(self.peak_clearance, clearance), self.peak_clearance)
        landed = self.in_jump & ~flight
        if self.train_peak:
            best = torch.where(landed & clean, self.peak_clearance, 0.).amax()
            torch.maximum(self.highest_landed_peak, best, out=self.highest_landed_peak)
        self.totals += torch.stack((takeoff.sum(), landed.sum(),
            (self.flight_ticks * self.dt * landed).sum(), (self.peak_clearance * landed).sum(),
            ((self.landing_ticks == 99) & contact.all(-1) & clean).sum()))
        self.landing_ticks = torch.where(landed & clean, 1,
            torch.where((self.landing_ticks > 0) & contact.all(-1) & clean, self.landing_ticks+1, 0))
        self.in_jump &= flight & clean
        self.flight_ticks *= self.in_jump
        self.peak_clearance *= self.in_jump

    def reward(self):
        contact, other = self.ground_contacts()
        supported = contact | self.last_contact
        self.last_contact.copy_(contact)
        flight = ~supported.any(-1)
        vz = self.robot.data.root_link_lin_vel_w[:, 2]
        delta = self.env.sim.data.xpos[:, self.wheel_ids] - self.env.sim.data.xanchor[:, self.hip_ids]
        quat = self.robot.data.root_link_quat_w[:, None].expand(-1, 2, -1)
        local = quat_apply_inverse(quat.reshape(-1, 4), delta.reshape(-1, 3)).reshape(-1, 2, 3)
        length = local[:, :, (0, 2)].norm(dim=-1)
        theta = torch.atan2(local[:, :, 0], -local[:, :, 2])
        vx_err = (self.linear[:, 0] - self.commands[:, 0]).square()
        yaw_err = (self.angular[:, 2] - self.commands[:, 1]).square()
        # 按上游实际乘法顺序：积分在腾空拍清零，并非普通的腾空计时器。
        first_contact = (self.air_integral > 0) & ~flight
        self.air_integral += self.dt * self.height.clamp(0., .50)
        encourage = .15 * ((self.air_integral - 5e-5) * first_contact + vz.clamp_min(0.))
        self.air_integral *= ~flight
        bottom = self.wheel_bottom()
        previous_peak = self.peak_clearance
        self.track_jumps(contact, flight, vz, other, bottom)
        raw = dict(tracking_lin_vel=2 * torch.exp(-vx_err/.25),
            tracking_lin_vel_enhance=2 * (torch.exp(-vx_err/2.5)-1),
            tracking_ang_vel=torch.exp(-yaw_err/.25), flight=flight.float(),
            encourage_jump=encourage, line_z=vz.clamp_min(0.) * flight,
            leg_tuck=torch.exp(-4*(length-TUCK).abs().sum(-1)) * flight,
            takeoff_extend=torch.exp(-4*(length-EXTEND).abs().sum(-1)) * (~flight & (vz > .15)),
            base_height_flight=torch.exp(-6*(self.height-FLIGHT_HEIGHT).abs()) * flight,
            nominal_state=(theta[:, 0]-theta[:, 1]).square()+10*(length[:, 0]-length[:, 1]).square(),
            pen_theta_no0=theta.square().sum(-1), orientation=self.gravity[:, :2].square().sum(-1),
            ang_vel_xy=self.angular[:, :2].square().sum(-1), torques=self.action.torques.square().sum(-1),
            collision=(other > .1).sum(-1),
            action_rate=(self.action.raw_action-self.previous[:, :, 0]).square().sum(-1))
        if self.landing is not None:
            if self.train_peak:
                # 检查策略拍内的每次物理接触，短时触地和反弹都不计高度进步。
                clean_flight = self.in_jump & ~self.landing.wheel_this_step.any(-1)
                clean_flight &= self.landing.body_this_step <= 1.
                peak_gain = (self.peak_clearance.square() - previous_peak.square()).clamp_min(0.)
            landing,descending,buffering = self.landing.rewards(contact,vz,length,bottom)
            raw.update(landing)
            raw['leg_tuck'] *= ~descending & ~buffering
            for name in ('flight','base_height_flight','line_z','encourage_jump','takeoff_extend'):
                raw[name] *= ~buffering
            if self.train_peak:
                raw['wheel_peak_gain'] = peak_gain * (clean_flight & ~buffering)
        terms = {name: (raw[name]*scale).clamp(-2.5, 2.5)*self.dt
                 for name, scale in self.cfg.rewards.scales.items()}
        return torch.stack(tuple(terms.values())).sum(0), terms
