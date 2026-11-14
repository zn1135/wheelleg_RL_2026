"""Frozen GPU inspection of PD200/P20D1 checkpoints, including saved training stages."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,timing_record
from ct_mjlab.fudan_ppo_walk.protocol import samples
from scripts.agent.fudan_ppo_walk import evaluate

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261113_pd200_p20d1_wheelP01_mechanical_delay10to20_noexternal_scratch3000_01'
EXPERIMENT=ROOT/'artifacts/fudan_ppo_walk/20261113-pd200-p20d1-delay10to20-scratch3000-01'
MODEL=RUN/'model_final.pt'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def items(*_):
    chosen=deepcopy(samples()[::21])
    for sample in chosen:
        if sample['case']['category']=='zero':
            sample['case']['duration']=30.;sample['case']['segments'][0]['duration']=30.
    return chosen


def run(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    metadata=torch.load(MODEL,map_location='cpu',weights_only=False)
    iteration=metadata['iterations']
    assert iteration>0 and not metadata['partial_update']
    resume=metadata.get('resume_from')
    log_start=0 if resume is None else resume['iterations']
    if resume is not None:
        restored=json.loads((EXPERIMENT/'initialization-audit.json').read_text())
        assert restored['passed'] and restored['source_checkpoint_sha256']==resume['sha256']
        assert restored['learner_parameters_exact'] and restored['both_optimizers_exact']
    assert metadata['transitions']==iteration*4096*48
    assert metadata['primary_updates']==metadata['encoder_updates']==iteration*20
    if MODEL.name=='model_final.pt':
        assert json.loads((EXPERIMENT/'training-audit.json').read_text())['passed']
        assert sha(MODEL)==json.loads((EXPERIMENT/'status.json').read_text())['checkpoint_sha256']
    else:
        assert MODEL.name==f'model_{iteration}.pt'
        assert json.loads((EXPERIMENT/'initialization-audit.json').read_text())['passed']
    source=json.loads((RUN/'source_sha256.json').read_text())
    source_check={name:sha(ROOT/name)==digest for name,digest in source.items()
        if name.startswith('mjlab_training/src/ct_mjlab/') or name.startswith('resources/')}
    assert all(source_check.values()),[name for name,passed in source_check.items() if not passed]
    conditions=from_record(metadata['disturbances']);assert not conditions.pushes
    assert conditions.external_force_range==conditions.external_torque_range==(0.,0.)
    plan=dict(checkpoint=str(MODEL),sha256=sha(MODEL),iteration=iteration,is_final_checkpoint=MODEL.name=='model_final.pt',samples=items(),
        resume_from=resume,log_start_iteration=log_start,
        control=metadata['control'],control_timing=metadata['control_timing'],disturbances=conditions.record(),
        runtime_source_matches_training=source_check,training_updates=0,
        scope='47 nominal cases with three zero-command cases extended to30s; no randomized acceptance population, get-up or real hardware')
    write(out/'plan.json',plan);shutil.copy2(__file__,out/'source.py')
    original=evaluate.configure_samples;recorded=[];ticks=[0];errors=torch.zeros(2,device='cuda:0')
    def configured(env,batch,folder):
        value=original(env,batch,folder);a=env.action
        assert (a.cfg.leg_stiffness,a.cfg.leg_damping,a.cfg.wheel_stiffness,a.cfg.wheel_damping)==(20.,1.,.1,0.)
        assert timing_record(env.cfg)==dict(physics_dt=.001,pd_dt=.005,policy_dt=.01)
        assert not a.wheel_motor.enabled and not a._wheel_speed_shaping and a.motor_strength.eq(1).all()
        for name,bounds in [('action_steps',conditions.action_delay_ms),('sensor_steps',conditions.sensor_delay_ms)]:
            low,high=[round(value/(1000*env.cfg.sim.dt)) for value in bounds]
            values=getattr(a.disturbances,name)
            assert values.min()>=low and values.max()<=high,name
        apply=a.apply_actions
        def observed_apply():
            update=a._pd_tick==0
            apply()
            if update:
                q=env.env.sim.data.qpos[:,a._motor_q_adr];v=env.env.sim.data.qvel[:,a._motor_v_adr]
                target=a._nominal_pos+.5*a._control_actions
                legs=a._leg_action_ids;wheels=a._wheel_action_ids
                target[:,legs]=torch.maximum(torch.minimum(target[:,legs],a._reference_pos[:,legs]+3.14),a._reference_pos[:,legs]-3.14)
                target[:,wheels]=(10*a._control_actions[:,wheels]).clamp(-150.,150.)
                expected=torch.zeros_like(q);expected[:,legs]=20*(target[:,legs]-q[:,legs])-v[:,legs]
                expected[:,wheels]=.1*(target[:,wheels]-v[:,wheels]);expected=expected.clamp(-a._effort_limits,a._effort_limits)
                errors[0]=torch.maximum(errors[0],(expected-a.torques).abs().max())
                errors[1]=torch.maximum(errors[1],env.env.sim.data.xfrc_applied.abs().max())
                recorded.append(torch.cat((target,q,v,a.torques),-1).clone())
            ticks[0]+=1
        a.apply_actions=observed_apply
        write(Path(folder)/'actual-control.json',dict(control=metadata['control'],control_timing=timing_record(env.cfg),
            physical_wheel_model=env.wheel_physics_profile,action_delay_steps=a.disturbances.action_steps.cpu().tolist(),
            sensor_delay_steps=a.disturbances.sensor_steps.cpu().tolist(),num_envs=env.num_envs))
        return value
    request=SimpleNamespace(checkpoint=MODEL,output=out/'gpu',deadline=time.time()+900,
        diagnostic=True,diagnostic_height=None,batch_size=47,selection=None,leg_load_profile='legacy',response_parameters=None)
    with patch.object(evaluate,'evaluation_items',items),patch.object(evaluate,'from_sim2sim_config',return_value=conditions),patch.object(evaluate,'configure_samples',configured),patch.object(FudanWalkCfg.sim,'wheel_model','mechanical'):
        evaluate.gpu(request)
    assert ticks[0]==30000 and len(recorded)==6000,(ticks,len(recorded))
    assert float(errors[0])<2e-5 and float(errors[1])==0.
    values=torch.stack(recorded).cpu().numpy()
    np.savez_compressed(out/'motor-pd200.npz',time=np.arange(len(values))*.005,values=values,
        fields=np.asarray(['target','actual','velocity','torque']),motor_order=np.asarray(['lf','lr','lw','rf','rr','rw']))
    write(out/'runtime-audit.json',dict(passed=True,integrations=ticks[0],pd_updates=len(recorded),
        PD_formula_max_error_Nm=float(errors[0]),external_wrench_peak=float(errors[1]),
        complete_traces=47,optimizer_updates=0,checkpoint_unchanged=sha(MODEL)==plan['sha256']))


def report(out):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads((out/'plan.json').read_text());rows=[]
    mapping=json.loads((out/'gpu/fixtures/mapping.json').read_text());legs=np.asarray(mapping['motor_q'])[[0,1,3,4]]
    for sample in plan['samples']:
        end=sample['case']['duration']
        with np.load(out/'gpu/traces'/(sample['id']+'.npz')) as tr:
            assert abs(tr['time'][-1]-end)<1e-8 and np.isfinite(tr['qpos']).all()
            last=(tr['time']>=end-5)&(tr['time']<end);one=(tr['qpos_time']>=end-1)&(tr['qpos_time']<end)
            five=(tr['qpos_time']>=end-5)&(tr['qpos_time']<end)
            w,x,y,z=tr['qpos'][:,3:7].T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
            rows.append(dict(sample_id=sample['id'],category=sample['case']['category'],height_command_m=sample['case']['segments'][-1]['command'][2],
                height_last5_mean_m=float(tr['position'][last,2].mean()),height_last5_min_m=float(tr['position'][last,2].min()),
                height_last5_max_m=float(tr['position'][last,2].max()),pitch_last5_peak_deg=float(abs(tr['roll_pitch_deg'][last,1]).max()),
                speed_last5_peak_m_s=float(np.linalg.norm(tr['linear_world'][last,:2],axis=1).max()),
                drift_last5_m=float(np.linalg.norm(tr['qpos'][five,:2]-tr['qpos'][five,:2][0],axis=1).max()),
                total_displacement_m=float(np.linalg.norm(tr['qpos'][-1,:2]-tr['qpos'][0,:2])),
                yaw_change_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
                body_contact_last5_mean_N=float(tr['base_contact'][last].mean()),wheel_support_last5_mean_N=tr['wheel_support'][last].mean(0).tolist(),
                leg_last1_max_pp_deg=float(np.ptp(np.rad2deg(tr['qpos'][one][:,legs]),axis=0).max()),
                leg_last5_max_pp_deg=float(np.ptp(np.rad2deg(tr['qpos'][five][:,legs]),axis=0).max())))
    event=EventAccumulator(str(RUN),size_guidance={'scalars':0}).Reload();reward=event.Scalars('Train/reward')
    reward=[v for v in reward if v.step<=plan['iteration']]
    y=np.asarray([v.value for v in reward]);steps=np.asarray([v.step for v in reward])
    assert np.array_equal(steps,np.arange(plan.get('log_start_iteration',0)+1,plan['iteration']+1))
    reward_stats=dict(first100_mean=float(y[:100].mean()),last100_mean=float(y[-100:].mean()),last300_mean=float(y[-300:].mean()),
        last300_std=float(y[-300:].std()),preceding300_mean=float(y[-600:-300].mean()),
        first_logged_iteration=int(steps[0]),last_iteration=int(steps[-1]))
    acceptance=json.loads((out/'acceptance/acceptance.json').read_text())
    score={r['sample_id']:r for r in acceptance['samples']}
    for r in rows:r.update(passed=score[r['sample_id']]['passed'],failure_reasons=score[r['sample_id']].get('failure_reasons',[]))
    write(out/'summary.json',dict(rows=rows,groups=acceptance['groups'],reward=reward_stats,
        checkpoint_iteration=plan['iteration'],is_final_checkpoint=plan['is_final_checkpoint'],
        technical_passed=acceptance['technical_passed'],total=47,optimizer_updates=0,behavior_accepted=False))
    fig,ax=plt.subplots(figsize=(9,3));ax.plot(steps,y,alpha=.2,lw=.6);ax.plot(steps[49:],np.convolve(y,np.ones(50)/50,mode='valid'),lw=1.5)
    ax.set(xlabel='Training iteration',ylabel='Mean reward',title=f'PD200 / P20 D1 / iteration {plan["iteration"]} / actual checkpoint recipe');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training-reward.png',dpi=150);plt.close(fig)
    motors=np.load(out/'motor-pd200.npz');mt=motors['time'];values=motors['values'];fig,axes=plt.subplots(3,2,figsize=(12,8))
    for row,height in enumerate((.22,.26,.30)):
        idx=next(i for i,s in enumerate(plan['samples']) if s['case']['category']=='zero' and s['case']['segments'][0]['command'][2]==height)
        # 左后腿与实录中最明显的往返运动对应；同时交付四腿最大范围指标。
        for col,window in enumerate(((0,30),(25,30))):
            select=(mt>=window[0])&(mt<window[1]);axes[row,col].plot(mt[select],np.rad2deg(values[select,idx,1]),label='Policy target',lw=.9)
            axes[row,col].plot(mt[select],np.rad2deg(values[select,idx,7]),label='Actual',lw=.9)
            axes[row,col].set(title=f'Zero command, height {height:.2f}m / left rear',ylabel='Joint angle (deg)',xlabel='Time (s)');axes[row,col].grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'zero-command-leg-tracking.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(passed=acceptance['technical_passed'],total=47,reward=reward_stats,zero=[r for r in rows if r['category']=='zero']),ensure_ascii=False))


def render(out,destination):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    plan=json.loads((out/'plan.json').read_text());summary=json.loads((out/'summary.json').read_text());destination.mkdir(parents=True,exist_ok=True)
    folder=out/'gpu';font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    groups=[(name,[s for s in plan['samples'] if s['case']['category']==name]) for name in ('zero','stop_forward','stop_backward')]
    failed=next((r for r in summary['rows'] if not r['passed'] and r['category'] not in ('zero','stop_forward','stop_backward')),None)
    if failed:groups.append(('failure_example',[s for s in plan['samples'] if s['id']==failed['sample_id']]))
    videos=[]
    delay=plan['disturbances']
    def delay_label(bounds):
        return f'{bounds[0]:g}' if bounds[0]==bounds[1] else f'{bounds[0]:g}-{bounds[1]:g}'
    for name,selection in groups:
        sides=[];duration=selection[0]['case']['duration'];count=len(selection)
        for sample in selection:
            model=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth=640;model.vis.global_.offheight=480;data=mujoco.MjData(model)
            cam=mujoco.MjvCamera();cam.distance=1.5;cam.elevation=-12.;cam.azimuth=90.
            option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
            with np.load(folder/'traces'/(sample['id']+'.npz')) as f:tr={k:f[k].copy() for k in f.files}
            sides.append((sample,model,data,cam,option,tr,mujoco.Renderer(model,width=640,height=480)))
        path=destination/(name+'.mp4');frames=0
        try:
            with imageio.get_writer(path,fps=50,codec='libx264',quality=7) as writer:
                for frame in range(round(duration*50)):
                    t=frame/50;canvas=Image.new('RGB',(640*count,544),'#18212b');draw=ImageDraw.Draw(canvas)
                    draw.text((12,7),f'1x NORMAL | iteration{plan["iteration"]} | PD200Hz P20/D1 | action{delay_label(delay["action_delay_ms"])}ms sensor{delay_label(delay["sensor_delay_ms"])}ms | {name} | t={t:.2f}s',font=font,fill='white')
                    draw.text((12,33),'Panels: target height0.22 /0.26 /0.30m. Camera follows XY; world displacement is printed.' if count==3 else 'Failure sample; camera follows XY, world displacement is printed.',font=font,fill='#ffcf7e')
                    for col,(sample,model,data,cam,option,tr,renderer) in enumerate(sides):
                        k=int(np.searchsorted(tr['qpos_time'],t-1e-8));tick=int(np.searchsorted(tr['time'],t-1e-8))
                        data.qpos[:]=tr['qpos'][k];mujoco.mj_forward(model,data);cam.lookat[:]=[data.qpos[0],data.qpos[1],.2]
                        renderer.update_scene(data,camera=cam,scene_option=option);panel=Image.fromarray(renderer.render().copy());pdraw=ImageDraw.Draw(panel)
                        pdraw.rectangle((0,0,640,110),fill='#18212b');command=evaluate.command(sample['case'],t)
                        lines=[f'cmd vx/yaw/h={command}',f'Actual h={tr["position"][tick,2]:.3f}m | pitch={tr["roll_pitch_deg"][tick,1]:+.1f}deg',
                            f'Speed={np.linalg.norm(tr["linear_world"][tick,:2]):.3f}m/s | XY from start={np.linalg.norm(tr["qpos"][k,:2]-tr["qpos"][0,:2]):.3f}m',
                            f'Wheel L/R={tr["wheel_support"][tick,0]:.1f}/{tr["wheel_support"][tick,1]:.1f}N | Body={tr["base_contact"][tick]:.1f}N']
                        for row,text in enumerate(lines):pdraw.text((10,5+25*row),text,font=font,fill='white')
                        canvas.paste(panel,(640*col,64))
                    writer.append_data(np.asarray(canvas));frames+=1
                    if frame==round((duration-1)*50):imageio.imwrite(destination/(name+'.png'),np.asarray(canvas))
        finally:
            for side in sides:side[-1].close()
        reader=imageio.get_reader(path);meta=reader.get_meta_data();decoded=sum(1 for _ in reader);reader.close()
        assert frames==decoded==round(duration*50) and meta['fps']==50
        videos.append(dict(file=str(path),frames=frames,fps=50,duration_s=duration,sha256=sha(path),full_decode_passed=True))
    for file in ('training-reward.png','zero-command-leg-tracking.png','summary.json'):
        shutil.copy2(out/file,destination/file)
    write(destination/'video-index.json',dict(videos=videos,physics_steps=0,model_forwards=0))
    write(out/'video-index.json',dict(videos=videos,physics_steps=0,model_forwards=0))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','report','render'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--destination',type=Path)
    p.add_argument('--checkpoint',type=Path,help='Saved complete model_N.pt or audited final checkpoint')
    p.add_argument('--experiment',type=Path,help='Training coordinator directory for the selected checkpoint')
    args=p.parse_args();out=args.output.resolve();torch.set_num_threads(1)
    global MODEL,RUN,EXPERIMENT
    if args.checkpoint is not None:MODEL=args.checkpoint.resolve();RUN=MODEL.parent
    if args.experiment is not None:EXPERIMENT=args.experiment.resolve()
    if args.mode=='run':run(out)
    elif args.mode=='report':report(out)
    else:render(out,args.destination.resolve())


if __name__=='__main__':main()
