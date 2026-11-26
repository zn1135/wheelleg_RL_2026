"""Phase rewards for landing. All extra contact state is local to each world."""

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg


def sample_landing(physics):
    state = getattr(physics, '_jump_landing', None)
    if state is not None:
        support = physics.scene['jump_wheel'].data.force[:, :, 2].abs() > 1.
        if state.preload:
            first = support.any(-1) & ~state.wheel_this_step.any(-1)
            tau = physics.sim.data.actuator_force[:,state.leg_motor_ids]
            state.first_drive = torch.where(first[:,None],state.extension(tau),state.first_drive)
        state.wheel_this_step |= support
        if state.stop_sink:
            state.dual_this_step |= support.all(-1)
        body = physics.scene['jump_ground'].data.force.norm(dim=-1).amax(-1)
        torch.maximum(state.body_this_step, body, out=state.body_this_step)
    return physics.sim.data.qpos[:, 0] * 0.


def install_landing_metrics(physical):
    physical.metrics['jump_landing_contacts'] = MetricsTermCfg(
        func=sample_landing, per_substep=True, reduce='max')


class Landing:
    def __init__(self, env, cfg):
        self.cfg = cfg
        self.env = env
        self.air = torch.zeros_like(env.episode_steps)
        self.buffer = torch.zeros_like(self.air)
        self.armed = torch.zeros_like(self.air, dtype=torch.bool)
        self.ascended = self.armed.clone()
        self.wheel_this_step = env.history.new_zeros((env.num_envs, 2), dtype=torch.bool)
        self.body_this_step = env.height.new_zeros(env.num_envs)
        self.stop_sink = cfg.get('stop_sink')
        if self.stop_sink:
            self.dual_this_step = self.armed.clone()
            self.loaded = self.armed.clone()
            self.base_mesh = env._collision_mesh('base_link')
        self.preload = cfg.get('preload')
        if self.preload:
            action = env.action
            assert not action.cfg.disturbances.action_delay and action.action_delay_steps.eq(0).all()
            assert env.cfg.control.decimation % env.cfg.control.pd_decimation == 0
            assert action._effort_limits[action._leg_action_ids].eq(40.).all()
            self.leg_motor_ids = [env.env.sim.mj_model.actuator('robot/'+n+'_motor').id
                                  for n in ('lf0','lf00','rf0','rf00')]
            self.first_drive = torch.zeros_like(self.wheel_this_step,dtype=env.height.dtype)
        env.env._jump_landing = self

    @staticmethod
    def extension(motors):
        # 当前闭链支路：左后髋减前髋、右前髋减后髋为伸腿方向。
        return torch.stack((motors[:,1]-motors[:,0],motors[:,2]-motors[:,3]),dim=1)

    def preload_rewards(self, descending, impact, waiting, saw_wheel, body_touch, vz, length, bottom):
        cfg = self.preload
        near = descending & (bottom.min(1).values <= -vz*cfg['lead_s']+cfg['height_margin_m'])
        early = waiting & saw_wheel & (self.buffer >= self.cfg['buffer_ticks']-1)
        a = self.env.action
        legs = a._leg_action_ids
        center = a._reference_pos[:,legs]
        # 本配方无延迟，策略拍末读取同一拍 PD 使用的限幅关节目标。
        target = (a._nominal_pos[:,legs]+a.cfg.leg_position_scale*a._control_actions[:,legs]).clamp(center-3.14,center+3.14)
        actual = self.env.env.sim.data.qpos[:,a._motor_q_adr[legs]]
        wanted = (self.extension(actual)+cfg['lead_diff_rad']).clamp(cfg['nominal_diff_rad'],cfg['max_diff_rad'])
        err = (self.extension(target)-wanted)/cfg['sigma_diff_rad']
        drive = self.extension(a._torques[:,legs]).amin(1)
        scale = cfg['support_diff_Nm']
        return dict(landing_motor_target=torch.exp(-err.square()).amin(1)*(near|early),
            landing_drive=(drive/scale).clamp(-1.,1.)*(near|early),
            landing_first_drive=(self.first_drive.amin(1)/scale).clamp(-1.,1.)*impact*saw_wheel*~body_touch,
            landing_early_length=(1.-(cfg['length_floor_m']-length.amin(1)).clamp_min(0.)/.05).clamp(-1.,1.)*early*~body_touch)


    def reset(self, ids):
        for state in (self.air, self.buffer, self.armed, self.ascended,
                      self.wheel_this_step, self.body_this_step):
            state[ids] = 0
        if self.preload:
            self.first_drive[ids] = 0
        if self.stop_sink:
            self.dual_this_step[ids] = False
            self.loaded[ids] = False

    def rewards(self, contact, vz, length, bottom):
        cfg = self.cfg
        saw_wheel = self.wheel_this_step.any(-1)
        body_touch = self.body_this_step > 1.
        self.armed |= contact.all(-1) & ~body_touch & (self.buffer==0)
        first_impact = (self.air>=cfg['air_ticks']) & self.ascended & (saw_wheel | body_touch)
        self.buffer = torch.where(first_impact, cfg['buffer_ticks'], (self.buffer-1).clamp_min(0))
        waiting = self.buffer>0
        unweighted = self.armed & ~saw_wheel & ~body_touch & ~waiting
        self.air = torch.where(unweighted, self.air+1, 0)
        self.ascended = unweighted & (self.ascended | (vz>.15))
        self.armed &= ~waiting & ~body_touch
        descending = (self.air>=cfg['air_ticks']) & self.ascended & (vz<0.)

        # 缓慢展开至两腿等长，触地后向有余量的长度压缩，再回到站高对应腿长。
        blend = (-vz/.8).clamp(0.,1.)
        reach = .23 + blend*(cfg['reach_m']-.23)
        elapsed = (cfg['buffer_ticks']-self.buffer)*self.env.dt
        compression = ((elapsed-.06).abs()/.06).clamp(0.,1.)
        stance_min = .23 if self.stop_sink else .18
        stance = (self.env.commands[:,2]-.035).clamp(stance_min,.27)
        recovery_length = cfg['compressed_m'] + compression*(
            torch.where(elapsed<.06, cfg['reach_m'], stance)-cfg['compressed_m'])
        pitch = torch.asin(self.env.gravity[:,0].clamp(-1.,1.))
        level = (bottom[:,0]-bottom[:,1])/cfg['wheel_level_sigma_m']
        upright = torch.exp(-(pitch/cfg['pitch_sigma_rad']).square())
        length_quality = lambda target: torch.exp(-((length-target[:,None])/cfg['length_sigma_m']).square().mean(-1))
        result = dict(
            landing_reach=length_quality(reach)*descending,
            landing_level=torch.exp(-level.square())*descending,
            landing_upright=upright*(descending | waiting),
            landing_compression=length_quality(recovery_length)*waiting*contact.any(-1)*~body_touch,
            landing_support=contact.all(-1)*~body_touch*waiting*upright*torch.exp(-(vz/.6).square()),
            landing_body=body_touch*waiting,
            landing_bounce=(vz.clamp_min(0.)/.8).square()*waiting)
        if self.stop_sink:
            # 只在实际双轮同时承重后开始计止沉，短暂离轮也不能把惩罚清掉。
            self.loaded = waiting & (self.loaded | self.dual_this_step)
            clearance = self.env.mesh_bottom(self.base_mesh)-self.env.env.scene.env_origins[:,2]
            shortage = (self.stop_sink['min_clearance_m']-clearance).clamp_min(0.)
            speed = (-vz).clamp_min(0.)
            result.update(
                landing_clearance=(shortage/self.stop_sink['clearance_width_m']).clamp_max(1.)*self.loaded,
                landing_sink=(speed/(speed+self.stop_sink['velocity_scale_m_s']))*self.loaded,
                landing_support_loss=(~contact.all(-1))*self.loaded)
            self.dual_this_step.zero_()
        if self.preload:
            result.update(self.preload_rewards(descending,first_impact,waiting,saw_wheel,body_touch,vz,length,bottom))
        self.wheel_this_step.zero_()
        self.body_this_step.zero_()
        return result, descending, waiting
