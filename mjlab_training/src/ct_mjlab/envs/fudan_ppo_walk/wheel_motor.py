"""Fudan-only wheel response after PD limits; controller gains and XML gain remain separate."""
import torch

from ct_mjlab.envs.chuanliantui.huananhu_action import wheel_response_input


class WheelMotorResponse:
    def __init__(self,action,sim_cfg):
        self.action=action;self.dt=action._env.physics_dt;self.n=action.num_envs
        self.mechanical=sim_cfg.wheel_model=='mechanical'
        self.worlds=torch.arange(self.n,device=action.device)
        self.configure(sim_cfg.wheel_motor_gain,sim_cfg.wheel_motor_delay_ms,
            sim_cfg.wheel_motor_time_constant_ms,sim_cfg.wheel_motor_torque_limit_nm)

    def configure(self,gain=1.,delay_ms=0.,time_constant_ms=0.,torque_limit_nm=None):
        a=self.action
        def column(value):return torch.as_tensor(value,device=a.device,dtype=a._torques.dtype).expand(self.n).clone()
        self.gain=column(gain);delay=column(delay_ms);self.tau_s=column(time_constant_ms)*.001
        if self.mechanical and ((self.gain!=1).any() or (delay!=0).any() or (self.tau_s!=0).any() or torque_limit_nm is not None):
            raise ValueError('Mechanical wheel plant cannot add output scaling, delay, lag or extra torque caps')
        if not all(torch.isfinite(x).all() and (x>=0).all() for x in (self.gain,delay,self.tau_s)):
            raise ValueError('Wheel gain/delay/time constant must be finite and nonnegative')
        self.delay_steps=torch.round(delay/(1000*self.dt)).long()
        self.alpha=-torch.expm1(-self.dt/self.tau_s.clamp(min=1e-12))
        self.limit=torch.full((self.n,2),float('inf'),device=a.device)
        if torque_limit_nm is not None:
            self.limit.copy_(torch.as_tensor(torque_limit_nm,device=a.device).expand(self.n,2))
            if not (self.limit>0).all() or torch.isnan(self.limit).any():
                raise ValueError('Wheel output torque limit must be positive shaft Nm or +inf/None for bypass')
        self.history=torch.zeros((int(self.delay_steps.max())+1,self.n,2),device=a.device)
        self.state=torch.zeros((self.n,2),device=a.device);self.head=0
        self.enabled=bool((self.gain!=1).any() or (self.delay_steps!=0).any() or (self.tau_s!=0).any() or torch.isfinite(self.limit).any())
        self.last_limited=torch.zeros_like(self.state);self.last_delayed=torch.zeros_like(self.state)
        self.last_ctrl=torch.zeros_like(self.state);self.last_shaft_torque=torch.zeros_like(self.state)

    def apply(self,limited,velocity):
        self.last_limited.copy_(limited)
        if self.enabled:
            self.history[self.head].copy_(limited)
            delayed=self.history[(self.head-self.delay_steps)%len(self.history),self.worlds]
            self.head=(self.head+1)%len(self.history)
            # tau=0逐值旁路，避免在无滞后候选中引入浮点相减误差。
            self.state.copy_(torch.where(self.tau_s[:,None]==0,delayed,
                self.state+self.alpha[:,None]*(delayed-self.state)))
            command=self.state*self.gain[:,None]
        else:
            delayed=limited;command=limited
        ctrl=wheel_response_input(command,velocity,self.action._wheel_response)
        xml_gain=self.action._wheel_response[:,0]
        shaft=ctrl*xml_gain
        clipped=shaft.clamp(-self.limit,self.limit)
        ctrl=torch.where(torch.isfinite(self.limit),clipped/xml_gain,ctrl)
        self.last_delayed.copy_(delayed);self.last_ctrl.copy_(ctrl);self.last_shaft_torque.copy_(clipped)
        return ctrl

    def reset(self,ids):
        self.history[:,ids]=0.;self.state[ids]=0.
        for field in (self.last_limited,self.last_delayed,self.last_ctrl,self.last_shaft_torque):field[ids]=0.

    def fixture(self):
        # -1表示没有额外输出上限，夹具中不保存非有限值。
        return dict(wheel_motor_gain=self.gain,wheel_motor_delay_steps=self.delay_steps,
            wheel_motor_tau_s=self.tau_s,wheel_motor_limit_nm=torch.where(torch.isfinite(self.limit),self.limit,-1.))
