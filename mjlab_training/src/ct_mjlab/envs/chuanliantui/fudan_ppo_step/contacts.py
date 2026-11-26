"""每1ms记录各部件最大接触，保留峰值位置，避免先合力再取模抵消。"""
import torch


class ContactMonitor:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        n, device = env.num_envs, env.device
        def zeros(*shape):
            return torch.zeros(n, *shape, device=device)
        self.ticks = zeros()
        self.count = zeros(2)
        self.streak = zeros(2)
        self.longest = zeros(2)
        self.peak = zeros(2)
        self.peak_pos = zeros(2, 3)
        self.any_streak = zeros()
        self.strong_streak = zeros()
        self.failed = zeros().bool()
        self.step_contact = zeros().bool()
        self.supported = zeros(2).bool()
        self.side_contact = zeros().bool()
        self.step_speed_peak = zeros()
        self.record_substeps = False
        self.step_forces = []
        self.step_positions = []
        self.total_ticks = 0
        self.capacity_overflow = zeros().bool()
        self.constraint_peak = torch.zeros((),device=device,dtype=torch.long)
        self.global_contact_peak = torch.zeros_like(self.constraint_peak)
        self._capacity_data = None
        if hasattr(env.sim, 'wp_data'):
            import warp as wp
            self._capacity_data = tuple(wp.to_torch(value) for value in (
                env.sim.wp_data.overflow,env.sim.wp_data.nefc,env.sim.wp_data.nacon))

    def reset(self, ids):
        for name in ('ticks', 'count', 'streak', 'longest', 'peak', 'peak_pos',
                     'any_streak', 'strong_streak', 'failed', 'step_contact', 'supported', 'side_contact', 'step_speed_peak'):
            getattr(self, name)[ids] = 0

    def update(self):
        self.total_ticks += 1
        if self._capacity_data is not None:
            overflow, constraints, contacts = self._capacity_data
            self.capacity_overflow |= overflow!=0
            self.constraint_peak = torch.maximum(self.constraint_peak,constraints.max())
            self.global_contact_peak = torch.maximum(self.global_contact_peak,contacts.max())
        self.step_speed_peak = torch.maximum(self.step_speed_peak,
            self.env.sim.data.qvel[:, :2].norm(dim=-1))
        dt = self.env.physics_dt
        forces, positions = [], []
        for name in ('step_base', 'step_legs'):
            data = self.env.scene[name].data
            norm = data.force.norm(dim=-1)
            peak, index = norm.max(dim=1)
            forces.append(peak)
            positions.append(data.pos[torch.arange(self.env.num_envs, device=self.env.device), index])
        force = torch.stack(forces, 1)
        pos = torch.stack(positions, 1)
        if self.record_substeps:
            self.step_forces.append(force.detach().clone())
            self.step_positions.append(pos.detach().clone())
        hot = force > self.cfg.contact_threshold_N
        self.ticks += 1
        self.count += hot
        self.streak = torch.where(hot, self.streak + dt, 0.)
        self.longest = torch.maximum(self.longest, self.streak)
        new_peak = force > self.peak
        self.peak_pos = torch.where(new_peak[..., None], pos, self.peak_pos)
        self.peak = torch.maximum(self.peak, force)
        self.step_contact |= hot.any(-1)
        self.any_streak = torch.where(hot.any(-1), self.any_streak + dt, 0.)
        robot_ids = self.env.scene['robot'].indexing.body_ids
        weight = self.env.sim.model.body_mass[:, robot_ids].sum(-1) * 9.81
        strong = force.amax(-1) > self.cfg.support_weight_fraction * weight
        self.strong_streak = torch.where(strong, self.strong_streak + dt, 0.)
        self.failed |= ((self.any_streak >= self.cfg.contact_timeout_s) |
                        (self.strong_streak >= self.cfg.support_timeout_s))
        wheels = self.env.scene['step_wheels'].data
        wheel_force = wheels.force.reshape(self.env.num_envs, 2, 4, 3)
        normal = wheels.normal.reshape(self.env.num_envs, 2, 4, 3)
        found = wheels.found.reshape(self.env.num_envs, 2, 4) > 0
        loaded = found & (wheel_force.norm(dim=-1) > self.cfg.contact_threshold_N)
        self.supported = (loaded & (normal[..., 2].abs() > .7)).any(-1)
        self.side_contact = (loaded & (normal[..., 2].abs() < .7)).any(dim=(1, 2))
        return force.amax(-1)

    def metrics(self, ids):
        return dict(contact_rate=(self.count[ids]/self.ticks[ids, None].clamp_min(1)).detach().clone(),
                    peak_N=self.peak[ids].detach().clone(),
                    longest_s=self.longest[ids].detach().clone(),
                    peak_position_w=self.peak_pos[ids].detach().clone())


def sample_contacts(env):
    if not hasattr(env, '_step_contacts'):
        return torch.zeros(env.num_envs, device=env.device)
    return env._step_contacts.update()
