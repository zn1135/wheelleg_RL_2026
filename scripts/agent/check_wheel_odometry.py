#!/usr/bin/env python3
"""Check signed encoder observations and run 20 teacher environments, without learning."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations as observations
from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.ppo import SmpCommandCfg, reset_pose
from ct_mjlab.wheel_odometry import WHEEL_ODOMETRY
from check_lqr_student_closed_loop import BridgeCfg, digest
from check_smp_policy_source import balance_substep


@torch.inference_mode()
def check(args):
    args.output.mkdir(parents=True, exist_ok=False)
    data = MotionDataset([args.dataset]);chosen = {}
    for episode in data.splits['train']:
        chosen.setdefault(episode['group_id'], episode)
    episodes = [chosen[k] for k in sorted(chosen)[:5]]
    assert len(episodes) == 5
    initial = []
    for episode in episodes:
        with np.load(episode['path']) as d:
            initial.append((d['initial_qpos'].copy(), d['initial_qvel'].copy()))
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    cfg.auto_reset=False;cfg.episode_length_s=21.
    cfg.commands['standup']=SmpCommandCfg()
    cfg.events['reset_pose']=EventTermCfg(func=reset_pose, mode='reset')
    cfg.metrics['source_balance']=MetricsTermCfg(func=balance_substep,per_substep=True,reduce='last')
    cfg.rewards={}
    cfg.actions['pd']=BridgeCfg(**{k:v for k,v in vars(cfg.actions['pd']).items() if k!='class_type'})
    cfg.actions['pd'].action_delay_s=None;cfg.actions['pd'].randomize_delays=False
    observations.configure_wheel_odometry(cfg)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=20,device=args.device)
    renderer=None
    try:
        raw=env.unwrapped;raw.reset(seed=42);action=raw.action_manager.get_term('pd')
        state=observations.get_state(raw);ids=torch.arange(20,device=raw.device)
        adr=state._motor_q_adr[4:6]
        origin=raw.sim.data.qpos[:,adr].clone()
        # 同轴向滚动为前进，反向为后退，两轮反向行驶为原地转向。
        increments=torch.tensor([[1.,-1.],[-1.,1.],[1.,1.],[20*np.pi,-20*np.pi]],device=raw.device).repeat(5,1)
        raw.sim.data.qpos[:,adr]=origin+increments;raw.sim.forward()
        expected=torch.tensor([.06,-.06,0.,1.2*np.pi],device=raw.device).repeat(5)[:,None]
        torch.testing.assert_close(state.wheel_travel(),expected,atol=3e-7,rtol=2e-6)
        q=raw.sim.data.qpos.clone();v=raw.sim.data.qvel.clone();fifo=action._action_history.clone()
        a=observations.wheel_actor_observation(raw);c=observations.wheel_critic_observation(raw)
        assert a.shape==(20,40) and c.shape==(20,83)
        torch.testing.assert_close(a[:,:39],state.actor(),atol=0,rtol=0)
        torch.testing.assert_close(c[:,:82],state.critic(),atol=0,rtol=0)
        torch.testing.assert_close(a[:,39:],expected,atol=3e-7,rtol=2e-6)
        torch.testing.assert_close(c[:,82:],a[:,39:],atol=0,rtol=0)
        torch.testing.assert_close(observations.wheel_actor_observation(raw),a,atol=0,rtol=0)
        assert torch.equal(q,raw.sim.data.qpos) and torch.equal(v,raw.sim.data.qvel)
        assert torch.equal(fifo,action._action_history)
        raw.sim.data.qpos[:,:2]+=10.;raw.sim.forward()
        torch.testing.assert_close(state.wheel_travel(),expected,atol=3e-7,rtol=2e-6)
        old_origin=state.wheel_origin.clone()
        observations.reset_observations(raw,ids[:1])
        assert state.wheel_travel()[0]==0
        torch.testing.assert_close(state.wheel_origin[1:],old_origin[1:],atol=0,rtol=0)
        torch.testing.assert_close(state.wheel_travel()[1:],expected[1:],atol=3e-7,rtol=2e-6)
        pairs=[(a,b) for a in (.002,.004) for b in (.002,.004)]
        raw.reset(seed=42)
        q=torch.tensor(np.array([q for pair in pairs for q,v in initial]),device=raw.device)
        v=torch.tensor(np.array([v for pair in pairs for q,v in initial]),device=raw.device)
        q[:,:2]=raw.scene.env_origins[:,:2]
        raw.sim.data.qpos.copy_(q);raw.sim.data.qvel.copy_(v);raw.sim.forward();action.reset()
        action.action_delay_steps[:]=torch.tensor([round(p[0]/.002) for p in pairs],device=raw.device).repeat_interleave(5)
        action.wheel_action_delay_steps[:]=torch.tensor([round(p[1]/.002) for p in pairs],device=raw.device).repeat_interleave(5)
        raw.command_manager.get_term('standup').ground_start[:]=True
        raw.command_manager.compute(dt=0.);observations.reset_observations(raw,ids)
        obs=raw.observation_manager.compute(update_history=True)
        assert not obs['actor'][:,39].any()
        raw._source_weight=float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        raw._source_probe=dict(live=torch.ones(20,dtype=torch.bool,device=raw.device),hold=torch.zeros(20,dtype=torch.long,device=raw.device),
            strict_run=torch.zeros(20,dtype=torch.long,device=raw.device),strict_ever=torch.zeros(20,dtype=torch.bool,device=raw.device))
        reader=MotionStateReader(raw)
        buffers={key:[] for key in ('qpos','qvel','observation','action','features','teacher_phase','done')}
        for tick in range(2000):
            inputs=obs['actor'].clone()
            # 独立公式核对本拍输入：使用初态轮角，不使用世界坐标或教师参考。
            angles=raw.sim.data.qpos[:,adr]
            reference=q[:,adr]
            distance=.03*((angles[:,0]-reference[:,0])-(angles[:,1]-reference[:,1]))
            torch.testing.assert_close(inputs[:,39],distance,atol=3e-7,rtol=2e-6)
            obs,_,done,_=env.step(torch.zeros(20,6,device=raw.device))
            assert not done.any(), 'Unexpected termination in teacher observation check'
            values=dict(qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,observation=inputs,action=action.raw_action,
                features=reader.features(),teacher_phase=action.teacher.controller.standup.phase,done=done)
            for key,value in values.items():buffers[key].append(value.cpu().numpy().copy())
            if tick<20:assert not raw.sim.data.actuator_force[:,action.teacher.motor_actuators].any()
            if tick%500==0:print(f'teacher check {tick+1}/2000',flush=True)
        arrays={key:np.array(value) for key,value in buffers.items()}
        np.savez_compressed(args.output/'trace.npz',**arrays,initial_qpos=q.cpu().numpy(),initial_qvel=v.cpu().numpy())
        assert np.isfinite(arrays['observation']).all() and np.isfinite(arrays['action']).all()
        # 视频只展示实际教师回放，不能当作新增40维策略已学会。
        movie=args.output/'wheel-observation.mp4';model=raw.sim.mj_model
        model.vis.global_.offwidth=960;model.vis.global_.offheight=540
        renderer=mujoco.Renderer(model,width=960,height=540);display=mujoco.MjData(model)
        camera=mujoco.MjvCamera();camera.azimuth=100;camera.elevation=-18;camera.distance=1.3
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',20)
        with imageio.get_writer(movie,fps=60,codec='libx264',quality=8,macro_block_size=1) as writer:
            for frame in range(1200):
                tick=min(1999,max(0,round(frame/60/.01)-1))
                display.qpos[:]=q[0].cpu().numpy() if frame==0 else arrays['qpos'][tick,0]
                display.qvel[:]=0 if frame==0 else arrays['qvel'][tick,0]
                camera.lookat[:]=display.qpos[:3];mujoco.mj_forward(model,display)
                renderer.update_scene(display,camera=camera)
                canvas=Image.fromarray(renderer.render());draw=ImageDraw.Draw(canvas)
                draw.rectangle((0,0,960,95),fill='#15202b')
                draw.text((10,5),'LQR教师实际回放｜新增轮式里程观测｜未训练新策略',font=font,fill='white')
                draw.text((10,34),f't={frame/60:.2f}s | 输入轮式里程={arrays["observation"][tick,0,39]:+.4f}m | actor40 / critic83',font=font,fill='white')
                draw.text((10,63),'从回合起点累计；不随停车、换命令或起立交接清零',font=font,fill='white')
                writer.append_data(np.asarray(canvas))
        with imageio.get_reader(movie) as video:
            assert video.count_frames()==1200
            imageio.imwrite(args.output/'preview.png',video.get_data(1080))
        result=dict(passed=True,optimizer_updates=0,num_envs=20,duration_s=20.,wheel_odometry=WHEEL_ODOMETRY,
            forward_reverse_turn_and_multiturn_checked=True,world_translation_invariant=True,
            repeated_reads_and_pd_fifo_unchanged=True,partial_reset_isolated=True,
            actor_first39_critic_first82_unchanged=True,live_pre_step_encoder_pairs_checked=True,
            startup_actual_force_zero=True,strict_rise=int(raw._source_probe['strict_ever'].sum()),
            final_balance=int((raw._source_probe['hold']>=500).sum()),
            video=dict(path=movie.name,sha256=digest(movie),frames=1200,fps=60),trace_sha256=digest(args.output/'trace.npz'),
            checker_sha256=digest(__file__),scope='Observation and teacher execution checks only; no trained40D policy or hardware acceptance')
        (args.output/'check.json').write_text(json.dumps(result,indent=2)+'\n')
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml',asdict(cfg));print(json.dumps(result,indent=2))
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');check(p.parse_args())
