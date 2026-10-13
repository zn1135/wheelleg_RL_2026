"""CPU manager rollout without autoreset, using the same r6 task kernels."""
import json
from pathlib import Path
import time
import torch
from .entry import RecoverVecEnv

def evaluate(actor,seconds=30,render=False,report=None,*,dynamic_modes=True,
             policy_joint_order=None,knee_limit_weight=None,reset_config=None,height_gate=None,
             command_config=None):
    from .model import POLICY_JOINTS
    env=RecoverVecEnv(num_envs=1,device='cpu',randomize=False,seed=42,auto_reset=False,
        dynamic_modes=dynamic_modes,policy_joint_order=policy_joint_order or POLICY_JOINTS,
        knee_limit_weight=knee_limit_weight,reset_config=reset_config,height_gate=height_gate,
        command_config=command_config)
    raw=env.unwrapped;state=raw.command_manager.get_term('standup')
    rows=[];first_failure=None;viewer=None
    if render:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(raw.sim.mj_model,raw.sim.mj_data)
    try:
        for i in range(round(seconds/raw.step_dt)):
            started=time.monotonic()
            with torch.inference_mode():actions=actor(env.obs) if actor else torch.zeros(1,6)
            _,_,done,_=env.step(actions)
            d=raw.scene['robot'].data
            rows.append(dict(time=(i+1)*raw.step_dt,height=float(d.root_link_pos_w[0,2]),
                command_height=float(state.command[0,2]),strict_recovered=bool(state.strict_recovered[0]),
                mode='recover' if raw._huananhu_observations.mode()[0,3]>.5 else 'normal',
                x=float(d.root_link_pos_w[0,0]),y=float(d.root_link_pos_w[0,1]),
                speed=float(d.root_link_lin_vel_b[0,:2].norm()),
                command_vx=float(state.command[0,0]),command_yaw=float(state.command[0,1]),
                velocity_error=float(torch.stack((
                    d.root_link_lin_vel_b[0,0]-state.command[0,0],
                    d.root_link_lin_vel_b[0,1])).norm())))
            if bool(raw.reset_terminated[0]) and first_failure is None:first_failure=rows[-1]['time']
            if not bool(torch.isfinite(raw.sim.data.qpos).all()):break
            if viewer:
                import mujoco
                if not viewer.is_running():break
                for name in ('qpos','qvel','ctrl'):
                    getattr(raw.sim.mj_data,name)[:]=getattr(raw.sim.data,name)[0].numpy()
                mujoco.mj_forward(raw.sim.mj_model,raw.sim.mj_data)
                viewer.sync();time.sleep(max(0,raw.step_dt-time.monotonic()+started))
            if bool(done[0]):break
    finally:
        if viewer:viewer.close()
        env.close()
    tail=rows[len(rows)//2:]
    metrics=dict(seconds=len(rows)*raw.step_dt,first_failure_s=first_failure,
        strict_recovered=bool(state.strict_recovered[0]),
        mean_height_last_half=sum(r['height'] for r in tail)/len(tail),
        max_height_error_last_half=max(abs(r['height']-r['command_height']) for r in tail),
        max_speed_last_half=max(r['speed'] for r in tail),
        max_velocity_error_last_half=max(r['velocity_error'] for r in tail))
    metrics['passed']=bool(first_failure is None and len(rows)*raw.step_dt>=20 and metrics['strict_recovered']
        and metrics['max_height_error_last_half']<=.03 and metrics['max_velocity_error_last_half']<=.05)
    if report:
        path=Path(report)
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(dict(metrics=metrics,trajectory=rows),indent=2)+'\n')
    return metrics
