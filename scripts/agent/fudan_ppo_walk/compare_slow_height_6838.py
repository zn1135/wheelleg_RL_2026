"""冻结6838、同初态GPU原始/0.5s/1s伸高对照；不操作实机或学习。"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import diagnose_gyro_x as core

ARMS=('fast','sham_fast','rise050','rise100','hold030')
SEEDS=(50001,50002,50003)


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def make_profiles(schedule):
    t=np.arange(3000)*.01
    indices=np.maximum(0,np.searchsorted(schedule[:,0],t+1e-8,side='right')-1)
    fast=schedule[indices,1]
    falling=np.flatnonzero((t>5)&(np.r_[0.,np.diff(fast)]< -1e-5))
    assert len(falling)
    low=float(fast[(t>t[falling[0]])&(t<10.)].min())
    lows=np.flatnonzero((t>=t[falling[0]])&(t<10.)&(abs(fast-low)<1e-7))
    start=int(lows[-1])
    end=int(np.flatnonzero((np.arange(len(t))>start)&(fast>=.299999))[0])
    assert .10 <= (end-start)*.01 <= .25 and np.allclose(fast[end:],.3,atol=1e-7)
    profiles=dict(fast=fast.copy(),sham_fast=fast.copy())
    ends=dict(fast=end,sham_fast=end)
    for name,ticks in (('rise050',50),('rise100',100)):
        value=fast.copy()
        value[start:start+ticks+1]=np.linspace(low,.3,ticks+1)
        value[start+ticks+1:]=.3
        profiles[name]=value;ends[name]=start+ticks
        assert np.array_equal(value[:start],fast[:start])
        assert np.all(np.diff(value[start:start+ticks+1])>=0) and value[start+ticks]==.3
    profiles['hold030']=fast.copy();profiles['hold030'][t>=2.]=.3;ends['hold030']=200
    for name,value in profiles.items():
        assert np.isfinite(value).all() and value.min()>=.22-1e-7 and value.max()<=.3+1e-7
    return t,profiles,dict(low_height_m=low,start_tick=start,start_s=start*.01,
        original_end_tick=end,original_duration_s=(end-start)*.01,
        end_tick=ends,end_s={k:v*.01 for k,v in ends.items()})


def prepare(out):
    core.ARMS=ARMS
    core.prepare(out,'slow_rise')
    schedule=np.asarray(json.loads((out/'height-schedule.json').read_text())['time_and_height'])
    t,profiles,event=make_profiles(schedule)
    np.savez_compressed(out/'planned-commands.npz',time=t,**profiles)
    plan=json.loads((out/'plan.json').read_text())
    for key in ('lowpass',):plan.pop(key,None)
    plan.update(arms=ARMS,study_kind='slow_rise',seeds=SEEDS,events=event,duration_s=30.,
        scope='15 nominal MuJoCo Warp GPU trajectories, three paired initial states; no hardware/training',
        intervention='Only the second rising height branch is time-stretched. Initial rise, dip and low hold unchanged. Maximum .30m.',
        nominal='Frozen6838 physical model, P20/D1,1/5/10ms, original dual10-20ms sampled delays, mu.8, no observation/PD noise or external load',
        transient='Original second rise plus0.50s/1.00s rises; same complete integration/PD/history initial states; fast/sham control; hold030 is a no-dip control',
        measurements=dict(relative_postrise_early_s=[0.,2.],relative_postrise_settled_s=[2.,7.],
            shared_tail_s=[25.,30.],rolling_spectral_windows='Ten1s windows after each rise;8-14Hz true gyroX band RMS',
            reproduction='Compare8-14Hz real recorded gyroX amplitude; no claim of curing real11Hz if fast baseline does not reproduce it'),
        acceptance='Only hold at23-30s is represented by scored case; common motion window25-30s. Actual full command arrays separately recorded.',
        hardware_operations=0)
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(__file__,out/'comparison-source.py')
    print(json.dumps(event),flush=True)


def worker(out):
    core.ARMS=ARMS
    plan=json.loads((out/'plan.json').read_text())
    with np.load(out/'planned-commands.npz') as z:
        planned=np.stack([z[a] for a in ARMS])
    captures=[];called=[]

    def selector(sample):
        if sample.get('diagnostic')!='height_replay':return False
        sample['case']['duration']=30.
        sample['case']['segments'][0]['duration']=30.
        return True

    def transform(samples,t,old):
        tick=round(t/.01)
        assert len(samples)==15
        commands=[[0.,0.,float(planned[ARMS.index(s['id'].split('__')[0]),tick])] for s in samples]
        called.append(commands)
        return commands

    def factory(policy,meta,n):
        import torch
        assert n==3
        class CheckedPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.policy=policy;self.tick=0;self.records=[];self.error=0.
                samples=[dict(id=f'height_replay__{s}',seed=s,randomized=False,
                    case=dict(id='post_rise_hold',category='zero',duration=30.,
                        segments=[dict(start=23.,duration=7.,command=[0.,0.,.3])])) for s in SEEDS]
                self.protocol_overrides={arm:dict(samples=samples,checkpoint_sha256=core.SHA,
                    scope='Diagnostic23-30s hold only, shared25-30s scoring window; actual full height schedule is external',
                    height_profile=arm,command_record_sha256=core.frozen.sha(out/'planned-commands.npz')) for arm in ARMS}

            def act_inference(self,obs,history):
                expected=obs.new_tensor(planned[:,self.tick]).repeat_interleave(n)*5
                self.error=max(self.error,float(abs(obs[:,8]-expected).max()))
                assert self.error<1e-6 and torch.equal(obs[:,8],history[:,-17])
                assert not obs[:,6:8].count_nonzero()
                self.records.append(torch.stack((obs[:,8]/5,history[:,-17]/5),-1).cpu().numpy())
                self.tick+=1
                return self.policy.act_inference(obs,history)
        result=CheckedPolicy();captures.append(result)
        return result

    core.worker(out,policy_factory=factory,sample_filter=selector,command_transform=transform)
    policy=captures[0]
    assert policy.tick==3000 and len(called)==3000
    np.savez_compressed(out/'delivered-commands.npz',commands=np.asarray(called),actor_and_history_height=np.asarray(policy.records))
    write(out/'command-audit.json',dict(passed=True,policy_ticks=policy.tick,actor_scaled_command_error=policy.error,
        actor_and_history_identical=True,hardware_operations=0,optimizer_updates=0,
        phases=plan['events'],command_record_sha256=core.frozen.sha(out/'delivered-commands.npz')))


def report(out):
    from scripts.agent.motion.assess import phase_verdict
    plan=json.loads((out/'plan.json').read_text());rows=[]
    for arm in ARMS:
        end=plan['events']['end_s'][arm]
        with np.load(out/arm/'motor-pd.npz') as z:pt=z['time'];pd=z['values']
        for i,seed in enumerate(SEEDS):
            sid=f'height_replay__{seed}'
            with np.load(out/arm/'traces'/(sid+'.npz')) as z:tr={k:z[k].copy() for k in z.files}
            complete=not bool(tr['terminated'].any() or tr['numerical_failure'].any())
            row=dict(arm=arm,seed=seed,rise_end_s=end,complete=complete,windows={})
            for name,lo,hi in (('early',end,end+2.),('settled',end+2.,end+7.),('tail',25.,30.)):
                take=(tr['time']>=lo)&(tr['time']<hi);m=(pt>=lo)&(pt<hi)
                row['windows'][name]=dict(time_s=[lo,hi],gyroX_band8to14_rad_s=float(core.band(tr['angular_body'][take,0],1000.)),
                    gyroX_rms_rad_s=float(np.sqrt(np.mean(tr['angular_body'][take,0]**2))),
                    gyroX_peak_rad_s=float(abs(tr['angular_body'][take,0]).max()),
                    leg_target_band8to14_rad=core.band(pd[m,i,0][:,[0,1,3,4]],200.).tolist(),
                    actual_leg_band8to14_rad=core.band(pd[m,i,1][:,[0,1,3,4]],200.).tolist(),
                    torque_band8to14_Nm=core.band(pd[m,i,3][:,[0,1,3,4]],200.).tolist(),
                    horizontal_speed_peak_m_s=float(np.linalg.norm(tr['linear_world'][take,:2],axis=-1).max()),
                    posture_peak_deg=float(abs(tr['roll_pitch_deg'][take]).max()))
            # 另报共同尾窗的统一运动门槛，不把振幅变小当作停车通过。
            take=(tr['time']>=25.)&(tr['time']<30.)
            row['tail_motion']=phase_verdict(vx=tr['linear_body'][:,0],yaw=tr['angular_body'][:,2],
                horizontal_speed=np.linalg.norm(tr['linear_world'][:,:2],axis=1),pitch_deg=tr['roll_pitch_deg'][:,1],
                vx_target=0.,yaw_target=0.,selected=take,valid=np.ones(len(take),bool),complete=complete,parking=True)
            row['postrise_1s_band8to14_rad_s']=[]
            for k in range(10):
                take=(tr['time']>=end+k)&(tr['time']<end+k+1)
                row['postrise_1s_band8to14_rad_s'].append(float(core.band(tr['angular_body'][take,0],1000.)))
            rows.append(row)
    groups=[]
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm]
        groups.append(dict(arm=arm,n=len(selected),complete=sum(r['complete'] for r in selected),
            tail_motion_passed=sum(r['tail_motion']['passed'] for r in selected),
            early_gyro_band_mean=float(np.mean([r['windows']['early']['gyroX_band8to14_rad_s'] for r in selected])),
            settled_gyro_band_mean=float(np.mean([r['windows']['settled']['gyroX_band8to14_rad_s'] for r in selected])),
            tail_gyro_band_mean=float(np.mean([r['windows']['tail']['gyroX_band8to14_rad_s'] for r in selected])),
            tail_speed_peak_mean=float(np.mean([r['windows']['tail']['horizontal_speed_peak_m_s'] for r in selected]))))
    sham=[]
    for seed in SEEDS:
        with np.load(out/'fast/traces'/f'height_replay__{seed}.npz') as a,np.load(out/'sham_fast/traces'/f'height_replay__{seed}.npz') as b:
            sham.append(dict(seed=seed,gyroX_max_difference=float(abs(a['angular_body'][:,0]-b['angular_body'][:,0]).max()),
                qpos_max_difference=float(abs(a['qpos']-b['qpos']).max())))
    write(out/'comparison.json',dict(rows=rows,groups=groups,sham=sham,optimizer_updates=0,hardware_operations=0,
        interpretation='No claim of curing real11Hz unless original fast rise first reproduces its strong oscillation.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(out/'planned-commands.npz') as z:t=z['time'];profiles={a:z[a] for a in ARMS}
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for arm in ('fast','rise050','rise100','hold030'):
        axes[0].plot(t,profiles[arm],label=arm)
        with np.load(out/arm/'traces/height_replay__50001.npz') as tr:
            axes[1].plot(tr['time'],tr['angular_body'][:,0],lw=.8,label=arm)
        with np.load(out/arm/'motor-pd.npz') as pd:
            axes[2].plot(pd['time'],pd['values'][:,0,0,1],lw=.8,label=arm)
    for ax,label in zip(axes,('Height command (m)','True gyroX (rad/s)','LR target (rad)')):
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8);ax.set_xlim(7.,18.)
    axes[-1].set_xlabel('Simulation seconds')
    fig.suptitle('Frozen6838 GPU | only second rise duration changes | same initial states | no sensor noise/external force')
    fig.tight_layout();fig.savefig(out/'slow-rise-comparison.png',dpi=140);plt.close(fig)
    height_response(out)
    print(json.dumps(groups),flush=True)


def height_response(out):
    plan=json.loads((out/'plan.json').read_text());rows=[];groups=[]
    for arm in ARMS:
        hi=plan['events']['end_s'][arm]+2.
        lo=plan['events']['start_s']
        # 无降高控制只供参考，仍使用快升相同的绝对时间窗。
        if arm=='hold030':hi=plan['events']['end_s']['fast']+2.
        for seed in SEEDS:
            path=out/arm/'traces'/f'height_replay__{seed}.npz'
            with np.load(path) as z:
                take=(z['time']>=lo)&(z['time']<hi)
                rows.append(dict(arm=arm,seed=seed,window_s=[lo,hi],
                    body_z_peak_m=float(z['position'][take,2].max()),
                    trajectory_sha256=core.frozen.sha(path)))
    for arm in ARMS:
        group=[r['body_z_peak_m'] for r in rows if r['arm']==arm]
        groups.append(dict(arm=arm,body_z_peak_mean_m=float(np.mean(group)),body_z_peak_max_m=max(group)))
    write(out/'height-response.json',dict(rows=rows,groups=groups,physics_steps=0,optimizer_updates=0,
        note='Supplementary height overshoot metric from the same saved traces; .30 is the commanded endpoint, not a hard cap on actual body height.'))
    print(json.dumps(groups),flush=True)


def verify(out):
    import ast
    plan=json.loads((out/'plan.json').read_text());runtime=json.loads((out/'runtime-audit.json').read_text())
    assert runtime['passed'] and runtime['checkpoint_unchanged'] and runtime['weights_unchanged']
    assert core.frozen.sha(out/'model_6838.pt')==core.SHA
    with np.load(out/'planned-commands.npz') as z:profiles=np.stack([z[a] for a in ARMS])
    with np.load(out/'delivered-commands.npz') as z:
        commands=z['commands'];actor=z['actor_and_history_height']
        expected=np.repeat(profiles.T,3,axis=1)
        assert np.array_equal(commands[:,:,2],expected) and not commands[:,:,:2].any()
        assert np.max(abs(actor[:,:,0]-expected))<1e-7 and np.array_equal(actor[:,:,0],actor[:,:,1])
        assert commands[:,:,2].max()<=.3000001
    arrays=0
    with np.load(out/'same-initial-state.npz') as z:
        for k in z.files:
            assert all(np.array_equal(z[k][0],other) for other in z[k][1:]),k
            arrays+=1
    for arm,ticks in (('rise050',50),('rise100',100)):
        begin=plan['events']['start_tick'];end=plan['events']['end_tick'][arm]
        assert end-begin==ticks
    result=json.loads((out/'comparison.json').read_text())
    assert all(r['complete'] for r in result['rows'])
    reference_path=core.ROOT/'artifacts/real2sim/20261125-high-length-onset6838-04/complete-cycle-162432.npz'
    with np.load(reference_path) as z:
        t=z['pd'][:,0]/1e6;names=list(z['physical_names']);g=z['physical'][:,names.index('gyro_x_rad_s')]
        selected=(t>=380.773242)&(t<=385.393247)
        reference=dict(source=str(reference_path),sha256=core.frozen.sha(reference_path),
            window_h7_s=[float(t[selected][0]),float(t[selected][-1])],
            gyroX_band8to14_rad_s=float(core.band(g[selected],200.)),
            gyroX_rms_rad_s=float(np.sqrt(np.mean(g[selected]**2))),
            limitation='Recorded strong-shaking window; not the same time window or matched physical initial state as the simulation.')
    null=[]
    for seed in SEEDS:
        with np.load(out/'fast/traces'/f'height_replay__{seed}.npz') as a,np.load(out/'sham_fast/traces'/f'height_replay__{seed}.npz') as b:
            null.append(dict(seed=seed,root_xyz_max_difference_m=float(abs(a['qpos'][:,:3]-b['qpos'][:,:3]).max()),
                gyroX_max_difference_rad_s=float(abs(a['angular_body'][:,0]-b['angular_body'][:,0]).max())))
    def function_ast(path,name):
        return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('worker','make_profiles'):
        assert function_ast(out/'comparison-source.py',name)==function_ast(__file__,name)
    assert function_ast(out/'diagnostic-source.py','worker')==function_ast(core.__file__,'worker')
    media=None
    if (out/'video/media.json').exists():
        import imageio.v2 as imageio
        media=json.loads((out/'video/media.json').read_text());path=out/'video/slow-rise-1x.mp4'
        assert core.frozen.sha(path)==media['video_sha256']
        with imageio.get_reader(path) as reader:
            assert reader.get_meta_data()['fps']==media['fps']
            assert sum(1 for _ in reader)==media['frames']
    shutil.copy2(__file__,out/'report-source.py')
    write(out/'verification.json',dict(passed=True,initial_arrays_equal=arrays,command_endpoint_and_duration=True,
        actual_actor_and_history_commands=True,weights_unchanged=True,source_worker_unchanged=True,
        video=media,real_reference=reference,sham_trajectory_difference=null,
        physics_steps=0,optimizer_updates=0,hardware_operations=0))
    write(out/'evidence-sha256.json',{str(p.relative_to(out)):core.frozen.sha(p) for p in sorted(out.rglob('*')) if p.is_file()})


def render(out):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image,ImageDraw
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    folder=out/'video';folder.mkdir(exist_ok=False)
    with np.load(out/'planned-commands.npz') as z:profiles={a:z[a] for a in ('fast','rise050','rise100')}
    panels=[]
    try:
        for arm in profiles:
            model=mujoco.MjModel.from_binary_path(str(out/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth,model.vis.global_.offheight=480,416
            data=mujoco.MjData(model);cam=mujoco.MjvCamera();cam.distance=1.6;cam.azimuth=100.;cam.elevation=-20.
            with np.load(out/arm/'traces/height_replay__50001.npz') as z:trace={k:z[k].copy() for k in z.files}
            panels.append((arm,model,data,cam,trace,mujoco.Renderer(model,width=480,height=416)))
        path=folder/'slow-rise-1x.mp4';fps=50;frames=400
        with imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=None) as writer:
            for f in range(frames):
                t=7.+f/fps;i=round(t/.01);j=round(t/.001);canvas=Image.new('RGB',(1440,416),'#18212b')
                for col,(arm,model,data,cam,tr,renderer) in enumerate(panels):
                    data.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                    cam.lookat[:]=data.qpos[:3]+[0.,0.,.03];renderer.update_scene(data,camera=cam)
                    image=Image.fromarray(renderer.render().copy());pen=ImageDraw.Draw(image)
                    pen.rectangle((0,0,480,72),fill='#18212b')
                    label={'fast':'Original fast (0.16s)','rise050':'Slow rise 0.50s','rise100':'Slow rise 1.00s'}[arm]
                    pen.text((8,5),f'6838 | {label} | 1x | t={t:.2f}s',fill='white')
                    pen.text((8,27),f'Height cmd={profiles[arm][i]:.4f}m | body z={tr["position"][j,2]:.4f}m',fill='white')
                    pen.text((8,49),f'TRUE gyroX={tr["angular_body"][j,0]:+.4f}rad/s | speed={np.linalg.norm(tr["linear_world"][j,:2]):.4f}m/s',fill='white')
                    canvas.paste(image,(col*480,0))
                writer.append_data(np.asarray(canvas))
                if f in (50,100,150,300):canvas.save(folder/f'key-{f:03d}.png')
        with imageio.get_reader(path) as reader:
            assert reader.get_meta_data()['fps']==fps and sum(1 for _ in reader)==frames
        write(folder/'media.json',dict(video_sha256=core.frozen.sha(path),fps=fps,frames=frames,duration_s=8.,
            simulation_time_s=[7.,15.],seed=50001,physics_steps=0,policy_forwards=0,
            trace_sha256={a:core.frozen.sha(out/a/'traces/height_replay__50001.npz') for a in profiles},
            note='SavedGPUposes only; representative initial seed, all3 seeds used for numeric comparison'))
    finally:
        for panel in panels:panel[-1].close()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','worker','report','render','verify'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    dict(prepare=prepare,worker=worker,report=report,render=render,verify=verify)[args.mode](args.output)


if __name__=='__main__':main()
