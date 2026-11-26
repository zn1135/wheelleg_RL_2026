"""冻结6838高位承受11Hz短力矩后撤扰，GPU闭环诊断；不训练/不操作实机。"""
import argparse
import ast
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from . import diagnose_gyro_x as core

ARMS=('control','sham_control','tx025','tx050','tx100')
AMPLITUDES=(0.,0.,.25,.5,1.)
SEEDS=(50001,50002,50003)
START,END,FREQUENCY=8.,8.5,11.


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def wave(t):
    x=np.asarray(t)-START;u=x/(END-START)
    return np.where((x>=0)&(x<END-START),np.sin(np.pi*u)**2*np.sin(2*np.pi*FREQUENCY*x),0.)


def prepare(out):
    core.ARMS=ARMS;core.prepare(out,'roll_burst')
    plan=json.loads((out/'plan.json').read_text());plan.pop('lowpass',None)
    plan.update(arms=ARMS,amplitudes_body_x_Nm=AMPLITUDES,seeds=SEEDS,
        scope='15 nominal MuJoCo Warp GPU high-stance perturbation trajectories; frozen6838, no hardware/training',
        intervention='Physical body-X roll torque at torso COM, rotated into world frame every1ms. No fake gyro signal.',
        nominal='P20/D1,1/5/10ms,mu.8,original dual10-20ms delays; no observation/PD noise or other external load',
        burst=dict(start_s=START,end_s=END,frequency_Hz=FREQUENCY,envelope='sin(pi*u)^2, smooth zero endpoints',
            formula='A*sin(pi*(t-8)/.5)^2*sin(2*pi*11*(t-8)) during[8,8.5); zero otherwise'),
        height='Original initial rise; hold0.30m from2s; vx/yaw0 throughout',duration_s=30.,
        measurements=dict(before_s=[7.5,8.],during_s=[8.,8.5],after_windows_s=[[8.5,9.],[9.,9.5],[9.5,10.5],[10.5,11.5]],
            recovery_criterion='Final post-withdrawal1s: true gyroX RMS<=.02rad/s, roll/pitch peaks<=5deg, common motion parking criterion speed peak<=.05m/s; diagnostic thresholds, not a full task acceptance',
            shared_acceptance_motion_window_s=[25.,30.]),
        transient=f'Peak body-X amplitudes{list(AMPLITUDES)}Nm, three copied initial states; sham control quantifies GPU divergence',
        reproduction='Passing injected-force recovery does not prove reproduction/cure of real self-excited11Hz',
        optimizer_updates=0,hardware_operations=0)
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    t=np.arange(30000)*.001;np.savez_compressed(out/'planned-torque.npz',time=t,unit_wave=wave(t),amplitudes=np.asarray(AMPLITUDES))
    shutil.copy2(__file__,out/'experiment-source.py')


def worker(out):
    core.ARMS=ARMS;before_rows=[];after_rows=[];begun=[False];restore=[]

    def selector(s):
        if s.get('diagnostic')!='height_replay':return False
        s['case']['duration']=s['case']['segments'][0]['duration']=30.
        return True

    def commands(samples,t,old):
        return [[0.,0.,.3 if t>=2. else cmd[2]] for cmd in old]

    def factory(policy,meta,n):
        import torch
        from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import fudan_ppo_walk as physical
        assert n==3
        original=physical.sample_substep
        # 同时记录积分后的外力端口，核对注入量没有被场景写入覆盖。
        def observed(physics_env):
            value=original(physics_env)
            if begun[0]:
                action=physics_env.action_manager.get_term('pd')
                after_rows.append(physics_env.sim.data.xfrc_applied[:,action._base_id].detach().clone())
            return value
        physical.sample_substep=observed;restore.append((physical,original))
        items=[dict(id=f'height_replay__{s}',seed=s,randomized=False,
            case=dict(id='post_burst_hold',category='zero',duration=30.,
                segments=[dict(start=23.,duration=7.,command=[0.,0.,.3])])) for s in SEEDS]
        policy.protocol_overrides={arm:dict(samples=items,checkpoint_sha256=core.SHA,
            scope='Only23-30s hold represented by scored case;25-30s motion window. Burst[8,8.5)s separately audited.',
            injected_roll_torque_peak_Nm=AMPLITUDES[i]) for i,arm in enumerate(ARMS)}
        return policy

    def apply(env,index):
        import torch
        from mjlab.utils.lab_api.math import quat_apply
        sim=env.env.sim;action=env.action;t=index*.001
        amplitude=sim.data.qpos.new_tensor(AMPLITUDES).repeat_interleave(3)
        local=sim.data.qpos.new_zeros((15,3));local[:,0]=amplitude*float(wave(t))
        quat=sim.data.qpos[:,3:7]
        sim.data.xfrc_applied.zero_()
        sim.data.xfrc_applied[:,action._base_id,3:]=quat_apply(quat,local)
        before_rows.append(torch.cat((sim.data.xfrc_applied[:,action._base_id],quat,local[:,:1]),1).detach().clone())
        begun[0]=True

    try:
        core.worker(out,policy_factory=factory,sample_filter=selector,command_transform=commands,physical_intervention=apply)
    finally:
        for module,original in restore:module.sample_substep=original
    import torch
    assert len(before_rows)==len(after_rows)==30000
    before=torch.stack(before_rows).cpu().numpy();after=torch.stack(after_rows).cpu().numpy()
    assert np.array_equal(before[:,:,:6],after)
    np.savez_compressed(out/'applied-torque.npz',time=np.arange(30000)*.001,
        before_integration=before,after_integration_wrench=after)
    actual_audit(out)


def actual_audit(out):
    with np.load(out/'applied-torque.npz') as z:t=z['time'];a=z['before_integration'];after=z['after_integration_wrench']
    expected=wave(t)[:,None]*np.repeat(AMPLITUDES,3)[None,:]
    assert np.array_equal(a[:,:,:6],after) and not after[:,:,:3].any()
    quat=a[:,:,6:10];rot=Rotation.from_quat(quat[:,:,[1,2,3,0]].reshape(-1,4))
    local=rot.inv().apply(after[:,:,3:].reshape(-1,3)).reshape(-1,15,3)
    error=float(abs(local[:,:,0]-expected).max());cross=float(abs(local[:,:,1:]).max())
    assert error<2e-6 and cross<2e-6
    outside=(t<START)|(t>=END)
    assert not after[outside].any()
    assert not after[:,:6].any()
    write(out/'torque-audit.json',dict(passed=True,substeps=len(t),body_axis_error_Nm=error,
        unwanted_body_yz_torque_Nm=cross,force_zero=True,world_force_port_preserved_through_integration=True,
        no_input_before_or_after_burst=True,body_torque_peak_Nm=float(abs(local[:,:,0]).max()),
        gyro_or_joint_feedback_override=False,optimizer_updates=0,hardware_operations=0))


def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))


def report(out):
    from scripts.agent.motion.assess import phase_verdict
    from .score_v2 import load_window
    rows=[]
    for ai,arm in enumerate(ARMS):
        with np.load(out/arm/'motor-pd.npz') as z:pt=z['time'];pd=z['values']
        for j,seed in enumerate(SEEDS):
            with np.load(out/arm/'traces'/f'height_replay__{seed}.npz') as z:tr={k:z[k].copy() for k in z.files}
            complete=not bool(tr['terminated'].any() or tr['numerical_failure'].any())
            row=dict(arm=arm,peak_torque_Nm=AMPLITUDES[ai],seed=seed,complete=complete,windows={})
            windows=[('before',7.5,8.),('during',8.,8.5),('after0',8.5,9.),('after1',9.,9.5),
                     ('after2',9.5,10.5),('after3',10.5,11.5),('tail',25.,30.)]
            for label,lo,hi in windows:
                m=(tr['time']>=lo)&(tr['time']<hi);pm=(pt>=lo)&(pt<hi)
                row['windows'][label]=dict(time_s=[lo,hi],gyroX_rms_rad_s=rms(tr['angular_body'][m,0]),
                    gyroX_peak_rad_s=float(abs(tr['angular_body'][m,0]).max()),
                    gyroX_band8to14_rad_s=float(core.band(tr['angular_body'][m,0],1000.)),
                    target_band8to14_rad=core.band(pd[pm,j,0][:,[0,1,3,4]],200.).tolist(),
                    actual_leg_band8to14_rad=core.band(pd[pm,j,1][:,[0,1,3,4]],200.).tolist(),
                    torque_band8to14_Nm=core.band(pd[pm,j,3][:,[0,1,3,4]],200.).tolist(),
                    speed_peak_m_s=float(np.linalg.norm(tr['linear_world'][m,:2],axis=1).max()),
                    roll_pitch_peak_deg=abs(tr['roll_pitch_deg'][m]).max(0).tolist())
            m=(tr['time']>=10.5)&(tr['time']<11.5)
            motion=phase_verdict(vx=tr['linear_body'][:,0],yaw=tr['angular_body'][:,2],
                horizontal_speed=np.linalg.norm(tr['linear_world'][:,:2],axis=1),pitch_deg=tr['roll_pitch_deg'][:,1],
                vx_target=0.,yaw_target=0.,selected=m,valid=np.ones(len(m),bool),complete=complete,parking=True)
            row['post3s_motion']=motion
            row['post3s_wheel_load']=load_window(tr,10500,11500,'postwithdrawal_final_second',dt=.001)
            row['recovered_at_3s']=bool(complete and motion['passed'] and row['windows']['after3']['gyroX_rms_rad_s']<=.02
                and max(row['windows']['after3']['roll_pitch_peak_deg'])<=5.
                and row['post3s_wheel_load']['passed'] and float(tr['base_contact'][m].max())<=1.)
            row['after_to_during_gyroX_rms_ratio']=row['windows']['after3']['gyroX_rms_rad_s']/row['windows']['during']['gyroX_rms_rad_s']
            rows.append(row)
    groups=[]
    for ai,arm in enumerate(ARMS):
        selected=[r for r in rows if r['arm']==arm]
        groups.append(dict(arm=arm,peak_torque_Nm=AMPLITUDES[ai],n=len(selected),recovered_at_3s=sum(r['recovered_at_3s'] for r in selected),
            gyroX_during_rms_mean=float(np.mean([r['windows']['during']['gyroX_rms_rad_s'] for r in selected])),
            gyroX_after_last1s_rms_mean=float(np.mean([r['windows']['after3']['gyroX_rms_rad_s'] for r in selected])),
            after_speed_peak_mean=float(np.mean([r['windows']['after3']['speed_peak_m_s'] for r in selected])),
            max_post3s_roll_pitch_deg=float(max(max(r['windows']['after3']['roll_pitch_peak_deg']) for r in selected))))
    write(out/'comparison.json',dict(rows=rows,groups=groups,optimizer_updates=0,hardware_operations=0,
        supplemental_recovery_gates='Existing wheel load v2 and no body contact in last postwithdrawal second; all thresholds reported separately',
        scope='Recovery from short injected physical torque, not reproduction or cure of real self-excited oscillation'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    with np.load(out/'applied-torque.npz') as z:t=z['time'];a=z['before_integration']
    gyro_peaks=[];target_limits=[];speed_peaks=[]
    for arm in ('control',*ARMS[2:]):
        i=ARMS.index(arm)*3
        axes[0].plot(t,a[:,i,10],lw=1,label=arm)
        with np.load(out/arm/'traces/height_replay__50001.npz') as z:
            axes[1].plot(z['time'],z['angular_body'][:,0],lw=.8,label=arm)
            axes[3].plot(z['time'],np.linalg.norm(z['linear_world'][:,:2],axis=1),lw=.8,label=arm)
            visible=(z['time']>=7.5)&(z['time']<12.)
            gyro_peaks.append(float(abs(z['angular_body'][visible,0]).max()))
            speed_peaks.append(float(np.linalg.norm(z['linear_world'][visible,:2],axis=1).max()))
        with np.load(out/arm/'motor-pd.npz') as z:
            axes[2].plot(z['time'],z['values'][:,0,0,1],lw=.8,label=arm)
            visible=(z['time']>=7.5)&(z['time']<12.)
            target_limits.extend([float(z['values'][visible,0,0,1].min()),float(z['values'][visible,0,0,1].max())])
    for ax,label in zip(axes,('Body-X torque (Nm)','True gyroX (rad/s)','LR target (rad)','Horizontal speed (m/s)')):
        ax.axvspan(START,END,color='grey',alpha=.15);ax.axvline(END,ls='--',color='black',alpha=.6)
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8);ax.set_xlim(7.5,12.)
    axes[3].axhline(.05,color='black',ls=':');axes[-1].set_xlabel('Simulation seconds; grey region=11Hz physical torque ON')
    limit=max(.02,1.15*max(gyro_peaks));axes[1].set_ylim(-limit,limit)
    axes[2].set_ylim(min(target_limits)-.01,max(target_limits)+.01)
    axes[3].set_ylim(0.,max(.06,1.15*max(speed_peaks)))
    fig.suptitle('Frozen6838 GPU | h_cmd=.30m | smooth0.5s torque burst then force exactlyzero | no learning/hardware')
    fig.tight_layout();fig.savefig(out/'withdrawal-response.png',dpi=140);plt.close(fig)
    print(json.dumps(groups),flush=True)


def render(out):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image,ImageDraw
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    folder=out/'video';folder.mkdir(exist_ok=False);panels=[]
    with np.load(out/'applied-torque.npz') as z:torque=z['before_integration'][:,:,10].copy()
    try:
        for arm in ('control',ARMS[3],ARMS[4]):
            model=mujoco.MjModel.from_binary_path(str(out/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth,model.vis.global_.offheight=480,416
            data=mujoco.MjData(model);cam=mujoco.MjvCamera();cam.distance=1.55;cam.azimuth=100.;cam.elevation=-20.
            with np.load(out/arm/'traces/height_replay__50001.npz') as z:tr={k:z[k].copy() for k in z.files}
            panels.append((arm,model,data,cam,tr,mujoco.Renderer(model,width=480,height=416)))
        path=folder/'burst-and-recovery-1x.mp4';fps=50;frames=250
        with imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=None) as writer:
            for f in range(frames):
                t=7.5+f/fps;i=round(t/.01);j=round(t/.001);canvas=Image.new('RGB',(1440,416),'#18212b')
                for col,(arm,model,data,cam,tr,renderer) in enumerate(panels):
                    data.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                    cam.lookat[:]=data.qpos[:3]+[0.,0.,.03];renderer.update_scene(data,camera=cam)
                    pic=Image.fromarray(renderer.render().copy());pen=ImageDraw.Draw(pic)
                    pen.rectangle((0,0,480,92),fill='#18212b')
                    peak=AMPLITUDES[ARMS.index(arm)];status='BEFORE' if t<START else '11Hz ON' if t<END else 'TORQUE REMOVED'
                    pen.text((8,5),f'6838 | peak {peak:g}Nm | {status} | 1x t={t:.2f}',fill='white')
                    pen.text((8,27),f'Body-X torque={torque[j,ARMS.index(arm)*3]:+.3f}Nm | cmd h=.30m',fill='white')
                    pen.text((8,49),f'TRUE gyroX={tr["angular_body"][j,0]:+.4f}rad/s',fill='white')
                    pen.text((8,71),f'Roll={tr["roll_pitch_deg"][j,0]:+.3f}deg | speed={np.linalg.norm(tr["linear_world"][j,:2]):.4f}m/s',fill='white')
                    canvas.paste(pic,(col*480,0))
                writer.append_data(np.asarray(canvas))
                if f in (40,60,100,200):canvas.save(folder/f'key-{f:03d}.png')
        with imageio.get_reader(path) as reader:assert reader.get_meta_data()['fps']==fps and sum(1 for _ in reader)==frames
        write(folder/'media.json',dict(video_sha256=core.frozen.sha(path),fps=fps,frames=frames,duration_s=5.,
            simulation_window_s=[7.5,12.5],seed=50001,physics_steps=0,policy_forwards=0,
            trace_sha256={p[0]:core.frozen.sha(out/p[0]/'traces/height_replay__50001.npz') for p in panels}))
    finally:
        for panel in panels:panel[-1].close()


def verify(out):
    import imageio.v2 as imageio
    audit=json.loads((out/'torque-audit.json').read_text());runtime=json.loads((out/'runtime-audit.json').read_text())
    assert audit['passed'] and runtime['passed'] and runtime['weights_unchanged'] and runtime['checkpoint_unchanged']
    assert core.frozen.sha(out/'model_6838.pt')==core.SHA and runtime['PD_error_nm']<2e-5
    def body(path,name):return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('worker','wave'):assert body(__file__,name)==body(out/'experiment-source.py',name)
    assert body(core.__file__,'worker')==body(out/'diagnostic-source.py','worker')
    with np.load(out/'same-initial-state.npz') as z:
        for k in z.files:assert all(np.array_equal(z[k][0],v) for v in z[k][1:]),k
    for arm in ARMS:
        with np.load(out/arm/'motor-pd.npz') as z:
            assert np.isfinite(z['values']).all()
            assert (abs(z['values'][:,:,3])<=np.array([40.,40.,3.9,40.,40.,3.9])+1e-4).all()
    with np.load(out/'feedback-record.npz') as z:
        a=z['values'];assert np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2])
    media=json.loads((out/'video/media.json').read_text());path=out/'video/burst-and-recovery-1x.mp4'
    assert core.frozen.sha(path)==media['video_sha256']
    with imageio.get_reader(path) as reader:assert reader.get_meta_data()['fps']==media['fps'] and sum(1 for _ in reader)==media['frames']
    for arm,digest in media['trace_sha256'].items():assert core.frozen.sha(out/arm/'traces/height_replay__50001.npz')==digest
    shutil.copy2(__file__,out/'delivery-source.py')
    write(out/'verification.json',dict(passed=True,actual_force_checked_every_ms=True,
        no_force_after_withdrawal=True,body_axis_transform=True,weights_unchanged=True,initial_arrays_equal=True,
        source_worker_unchanged=True,video_full_decode=True,physics_steps=0,optimizer_updates=0,hardware_operations=0))
    write(out/'evidence-sha256.json',{str(p.relative_to(out)):core.frozen.sha(p) for p in sorted(out.rglob('*')) if p.is_file()})


def main():
    global ARMS,AMPLITUDES
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','worker','report','render','verify'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--amplitudes',type=float,nargs=3,default=(.25,.5,1.),help='仅prepare使用的三档峰值Nm，后续步骤读取plan')
    a=p.parse_args()
    if a.mode=='prepare':
        if not all(np.isfinite(x) and 0<x<=10. for x in a.amplitudes) or list(a.amplitudes)!=sorted(set(a.amplitudes)):
            p.error('Use three increasing finite amplitudes in(0,10]Nm')
        AMPLITUDES=(0.,0.,*a.amplitudes)
        ARMS=('control','sham_control',*(f'tx{round(x*100):03d}' for x in a.amplitudes))
    else:
        plan=json.loads((a.output/'plan.json').read_text())
        ARMS=tuple(plan['arms']);AMPLITUDES=tuple(plan['amplitudes_body_x_Nm'])
    dict(prepare=prepare,worker=worker,report=report,render=render,verify=verify)[a.mode](a.output)


if __name__=='__main__':main()
