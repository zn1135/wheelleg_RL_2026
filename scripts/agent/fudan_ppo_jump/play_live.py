"""Keyboard-operated Jump5200 on MuJoCo Warp with a local MuJoCo viewer."""
import argparse
import json
from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer
from mujoco.glfw import glfw
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import TRIGGER, jump_config
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from .evaluate import SEED, bar_distance, policies
from .handoff import LandingHandoff

CHECKPOINT = Path('logs/mjlab/fudan_ppo_jump/20261119_preload4200_to5200_4096_01/model_final.pt')
SCENES = (('Flat', None), ('Bar 5 cm', .05), ('Bar 10 cm', .10))
DRIVE_SPEED = 3.
TOGGLES = (glfw.KEY_J, glfw.KEY_T, glfw.KEY_R, glfw.KEY_P, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3)


@torch.inference_mode()
def play_scene(walk, jump, scene, log, end_at):
    label,height=SCENES[scene]
    cfg=jump_config()
    cfg.env.episode_length_s=3600.
    env=JumpEnv(1,SEED,'cuda:0',False,cfg=cfg,bar_height=height,record_flat=height is None)
    started=torch.full((1,),-1,device=env.device)
    model=env.env.sim.mj_model
    view_state=mujoco.MjData(model)
    window=None
    def get_window(_):
        nonlocal window
        window=glfw.get_current_context()
    def reset():
        torch.manual_seed(SEED)
        env.reset_idx(torch.arange(1,device=env.device))
        env.commands[:]=env.commands.new_tensor((0.,0.,.26))
        obs,_=env.observations()
        env.history.copy_(obs.repeat(1,5))
        env._history_needs_fill.zero_()
        started.fill_(-1)
        return LandingHandoff(started,8)
    gate=reset()
    view_state.qpos[:]=env.env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(model,view_state)
    before=set(threading.enumerate())
    handle=mujoco.viewer.launch_passive(model,view_state,key_callback=get_window,
                                        show_left_ui=False,show_right_ui=False)
    viewer_thread=next((t for t in set(threading.enumerate())-before if '_launch_internal' in t.name),None)
    def write(event,**values):
        print(json.dumps(dict(event=event,scene=label,**values)),file=log,flush=True)
    write('opened',pd_ms=5,physics_ms=1,policy_ms=10,buffer_ms=80,drive_speed_m_s=DRIVE_SPEED)
    selection=None
    armed=False
    paused=False
    fallen=False
    toggle_last=set()
    tick=0
    last_jump=False
    last_command=(0.,0.)
    next_frame=time.monotonic()
    deadline=next_frame
    try:
        with handle.lock():
            handle.cam.azimuth=105.
            handle.cam.elevation=-16.
            handle.cam.distance=2.5
        while handle.is_running() and (end_at is None or time.monotonic()<end_at):
            def held(key):
                return window is not None and glfw.get_key(window,key)==glfw.PRESS

            pressed={k for k in TOGGLES if held(k)}
            new=pressed-toggle_last
            toggle_last=pressed
            change=next((i for i,k in enumerate((glfw.KEY_1,glfw.KEY_2,glfw.KEY_3)) if k in new),None)
            if change is not None and change!=scene:
                selection=change
                break
            if glfw.KEY_R in new:
                gate=reset()
                armed=paused=fallen=last_jump=False
                tick=0
                write('reset')
            if glfw.KEY_P in new:
                paused=not paused
            active=bool(gate.active(tick).item())
            if glfw.KEY_T in new and not active and height is not None:
                armed=not armed
                write('armed',on=armed)
            if not paused and not fallen:
                speed=DRIVE_SPEED*(int(held(glfw.KEY_UP))-int(held(glfw.KEY_DOWN)))
                yaw=1.*(int(held(glfw.KEY_LEFT))-int(held(glfw.KEY_RIGHT)))
                if held(glfw.KEY_X):
                    speed=yaw=0.
                if armed or glfw.KEY_J in new:
                    distance=float(bar_distance(env)[0]) if height is not None else -1.
                    if not active and (glfw.KEY_J in new or (speed>0 and 0<distance<TRIGGER)):
                        started.fill_(tick)
                        gate=LandingHandoff(started,8)
                        active=True
                        armed=False
                if (speed,yaw)!=last_command:
                    write('command',tick=tick,vx=speed,yaw=yaw)
                    last_command=(speed,yaw)
                env.commands[:]=env.commands.new_tensor((speed,yaw,.26))
                obs,_=env.observations()
                env.history[:,106:109]=obs[:,6:9]
                if active:
                    action,_=jump.act_inference(obs,env.history)
                else:
                    obs[:,6]*=2/3
                    walk_history=env.history.reshape(1,5,25).clone()
                    walk_history[:,:,6]*=2/3
                    action,_=walk.act_inference(obs,walk_history.flatten(1))
                if active!=last_jump:
                    write('jump' if active else 'walk',tick=tick,x=float(env.position[0,0]),
                          touchdown=int(gate.touchdown.item()))
                    last_jump=active
                env.substeps.clear()
                _,_,_,done,_,_=env.step(action,auto_reset=False,resample=False)
                assert len(env.substeps)==10
                if env.env.sim.wp_data.overflow.numpy().any():
                    raise RuntimeError('MuJoCo Warp contact or constraint storage overflow')
                if active:
                    gate.observe(tick,torch.stack(env.substeps),gate.active(tick))
                fallen=bool(done.item())
                tick+=1
                if tick%50==0:
                    write('state',tick=tick,x=float(env.position[0,0]),z=float(env.position[0,2]),
                          vx=float(env.linear[0,0]))
                if fallen:
                    write('fell',tick=tick)
            now=time.monotonic()
            if now>=next_frame:
                with handle.lock():
                    view_state.qpos[:]=env.env.sim.data.qpos[0].cpu().numpy()
                    view_state.time=tick*.01
                    mujoco.mj_forward(model,view_state)
                    handle.cam.lookat[:]=view_state.qpos[:3]+[.3,0,.06]
                mode='FALLEN - R to reset' if fallen else 'PAUSED' if paused else 'JUMP' if active else 'WALK'
                handle.set_texts((None,None,
                    f'Arrows: +/-{DRIVE_SPEED:g} m/s / steer  |  J: jump now  |  T: arm bar trigger\n'
                    '1: flat  2: bar 5 cm  3: bar 10 cm  |  R: reset  P: pause  X: stop',
                    f'{label} / {mode}\n{"ARMED" if armed else ""}  vx {float(env.commands[0,0]):+.1f}  yaw {float(env.commands[0,1]):+.1f}\n{tick*.01:.2f} s'))
                handle.sync(state_only=True)
                next_frame=now+.033
            deadline=max(deadline+.01,time.monotonic())
            time.sleep(max(0.,deadline-time.monotonic()))
    finally:
        handle.close()
        if viewer_thread is not None:
            viewer_thread.join(timeout=3.)
        env.close()
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    parser.add_argument('--scene',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--events',type=Path,required=True)
    args=parser.parse_args()
    walk,jump,saved=policies(args.checkpoint,'cuda:0')
    assert saved['iterations']==5200
    end_at=time.monotonic()+args.seconds if args.seconds is not None else None
    scene=args.scene-1
    with args.events.open('x') as log:
        while scene is not None:
            scene=play_scene(walk,jump,scene,log,end_at)


if __name__=='__main__':
    main()
