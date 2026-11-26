"""Frozen6838 GPU gyro-X comparison: raw/sham/half-gain/10Hz low-pass, no learning."""
import argparse
from copy import deepcopy
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.signal import periodogram

from . import inspect_randomization_4000 as frozen
from . import inspect_slope5to20_mu08 as study
from .run_torque_ablation import run_command

ROOT = frozen.ROOT
MODULE = 'scripts.agent.fudan_ppo_walk.diagnose_gyro_x'
ARMS = ('raw', 'sham', 'gain_half', 'lowpass10')
SHA = '8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad'
REAL = ROOT/'artifacts/real2sim/20261117-high-length6838-01/raw/高腿长抖腿/6838高腿长对照_162432/full-pd.csv'


def prepare(out, study_kind='gyro'):
    assert frozen.sha(study.RUN/'model_final.pt') == SHA
    hashes = json.loads((study.RUN/'source_sha256.json').read_text())
    checks = {}
    for name, digest in hashes.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            checks[name] = frozen.sha(study.RUN/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/')) == digest
        elif name.startswith('resources/'):
            checks[name] = frozen.sha(ROOT/name) == digest
    assert all(checks.values())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(study.RUN/'model_final.pt', out/'model_6838.pt')
    shutil.copy2(__file__, out/'diagnostic-source.py')
    with REAL.open() as f:
        rows = [r for r in csv.DictReader(f) if r['rl_engaged']=='1' and r['warmup']=='0' and r['pd_valid']=='1']
    first = float(rows[0]['h7_us'])
    unique = [r for i,r in enumerate(rows) if i==0 or r['obs_sequence']!=rows[i-1]['obs_sequence']]
    schedule = [(float(r['h7_us'])/1e6-first/1e6, float(r['obs_8'])/5) for r in unique]
    frozen.write(out/'height-schedule.json', dict(time_and_height=schedule, source=str(REAL), sha256=frozen.sha(REAL)))
    frozen.write(out/'plan.json', dict(arms=ARMS, study_kind=study_kind, optimizer_updates=0, training_started=False,
        checkpoint_sha256=SHA, frozen_source_checks=checks,
        scope='47 nominal regressions plus recorded-height and high-hold routes at3 paired initial seeds',
        intervention=('Only delayed gyroX in Actor current frame and its five-frame history; PD velocity feedback unchanged'
            if study_kind=='gyro' else 'Only gravity projection rotated aboutY by0/-2/+2deg in current frame and five-frame history; gyro and physical state unchanged'),
        lowpass=dict(cutoff_Hz=10., policy_dt=.01, alpha=1-np.exp(-2*np.pi*10*.01)),
        nominal='Original47 cases, zero commands extended to30s; friction0.8, original dual10-20ms delays, no external loads',
        transient='15s zero-motion replay of supplied height command and paired .30m-hold control; no injected IMU noise or forced11Hz movement',
        guard='Identical integration/controller/delay/history initial states copied across all arms; raw-sham quantifies GPU divergence',
        reproduction='If raw GPU does not show measured11Hz amplitude, do not claim real shaking cured', wall_limit_s=900))


def worker(out, policy_factory=None, sample_filter=None):
    import torch
    frozen.RUN = study.RUN
    assert not any(n=='ct_mjlab' or n.startswith('ct_mjlab.') for n in sys.modules)
    sys.meta_path.insert(0, frozen.FrozenTrainingSource())
    import mujoco
    from mjlab.managers.event_manager import RecomputeLevel
    from ct_mjlab.fudan_ppo_walk.protocol import samples
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    from ct_mjlab.fudan_ppo_walk.measurement import configure_samples, update_evaluation_commands, gpu_row, arrays
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, timing_record
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import from_record
    from scripts.agent.motion.acceptance import canonical
    torch.set_num_threads(1)
    chosen = deepcopy(samples()[::21])
    for s in chosen:
        if s['case']['category']=='zero':
            s['case']['duration']=s['case']['segments'][0]['duration']=30.
    nominal = deepcopy(chosen)
    for kind in ('height_replay', 'height_hold'):
        for seed in (50001, 50002, 50003):
            chosen.append(dict(id=f'{kind}__{seed}', seed=seed, randomized=False,
                case=dict(id='paired_height_transient', category='zero', duration=15.,
                    segments=[dict(start=0., duration=15., command=[0.,0.,.22])]), diagnostic=kind))
    if sample_filter is not None:
        chosen = [s for s in chosen if sample_filter(s)]
        kept = {s['id'] for s in chosen}
        nominal = [s for s in nominal if s['id'] in kept]
    n = len(chosen)
    all_samples = [dict(s, id=arm+'__'+s['id']) for arm in ARMS for s in chosen]
    policy, meta = load_policy(out/'model_6838.pt', 'cuda:0')
    if policy_factory is not None:policy = policy_factory(policy, meta, n)
    study_kind=json.loads((out/'plan.json').read_text()).get('study_kind','gyro')
    before = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    cfg = FudanWalkCfg()
    recipe = json.loads((study.RUN/'effective-training-config.json').read_text())
    for section, fields in recipe.items():
        for key,value in fields.items(): setattr(getattr(cfg,section), key, value)
    cfg.terrain.mode='plane'
    cfg.observations.critic_height_scan=False
    conditions=replace(from_record(meta['disturbances']), pushes=False,
        external_force_range=(0.,0.), external_torque_range=(0.,0.))
    original_cfg=physical.make_env_cfg
    def make_cfg(*args,**kwargs):
        c=original_cfg(*args,**kwargs);c.sim.nconmax,c.sim.njmax=64,256;return c
    with patch.object(physical,'make_env_cfg',make_cfg):
        env=physical.FudanEnv(len(all_samples),seed=50001,randomized=True,cfg=cfg,
            reset_contract=meta['reset_contract'],disturbances=conditions)
    sim=env.env.sim;action=env.action
    assert timing_record(cfg)==dict(physics_dt=.001,pd_dt=.005,policy_dt=.01)
    fixture=out/'fixtures'
    obs,_=configure_samples(env,all_samples,fixture)
    model=sim.mj_model
    terrain=np.flatnonzero(model.geom_bodyid==model.body('terrain').id)
    bodies=env.robot.indexing.body_ids.cpu().numpy()
    robot=np.flatnonzero(np.isin(model.geom_bodyid,bodies))
    geoms=np.union1d(terrain,robot).tolist()
    sim.model.geom_friction[:,geoms,0]=.8;env.env._fudan_parameters['friction'].fill_(.8)
    sim.recompute_constants(RecomputeLevel.set_const)
    model.geom_friction[geoms,0]=.8;mujoco.mj_saveModel(model,str(fixture/'model.mjb'))
    actual_material=sim.model.geom_friction.cpu().numpy()
    for i,s in enumerate(all_samples):
        file=fixture/(s['id']+'_initial.npz')
        with np.load(file) as saved: fields={k:saved[k].copy() for k in saved.files}
        fields['geom_friction']=actual_material[i];np.savez_compressed(file,**fields)
    # 同一批各分支复制完整控制/积分初态，避免独立GPU运行的起点差异。
    source=torch.arange(n,device=env.device).repeat(len(ARMS)-1)
    dest=torch.arange(n,len(all_samples),device=env.device)
    env.env.scene.env_origins[dest]=env.env.scene.env_origins[source].clone()
    saved={}
    for name in ('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','time','eq_active'):
        value=getattr(sim.data,name);value[dest]=value[source].clone();saved['physics.'+name]=value
    owners=[('wrapper',env),('pd',action),('delay',action.disturbances),('motor',action.wheel_motor)]
    for name in ('wheel_contact','base_contact'):
        sensor=env.env.scene[name];owners.extend([(name,sensor),(name+'.data',sensor.data)])
    for label,owner in owners:
        for name,value in vars(owner).items():
            if not isinstance(value,torch.Tensor) or name in ('ids','_env_ids','env_ids'):continue
            if name in ('_action_history','actions','sensors') and value.ndim>1 and value.shape[1]==len(all_samples):
                value[:,dest]=value[:,source].clone();saved[label+'.'+name]=value.transpose(0,1)
            elif value.ndim and value.shape[0]==len(all_samples):
                value[dest]=value[source].clone();saved[label+'.'+name]=value
    obs[dest]=obs[source].clone();saved['actor.obs']=obs
    sim.forward();env.env.scene.update(dt=0.);sim.sense()
    paired={}
    for key,value in saved.items():
        a=value.cpu().numpy().copy().reshape(len(ARMS),n,*value.shape[1:])
        assert all(np.array_equal(a[0],v) for v in a[1:]),key
        paired[key]=a
    np.savez_compressed(out/'same-initial-state.npz',**paired)
    # 夹具必须记录实际复制后的初态；先前逐世界导出的nominal存在1ULP差异。
    initial_before=out/'initial-export-before-fork';initial_before.mkdir()
    for ai,arm in enumerate(ARMS):
        for i,s in enumerate(chosen):
            file=fixture/(arm+'__'+s['id']+'_initial.npz')
            shutil.copy2(file,initial_before/file.name)
            with np.load(file) as d:fields={k:d[k].copy() for k in d.files}
            fields['qpos']=paired['physics.qpos'][ai,i].copy();fields['qpos'][:2]=0.
            fields['qvel']=paired['physics.qvel'][ai,i].copy()
            fields['nominal']=paired['pd._nominal_pos'][ai,i].copy()
            np.savez_compressed(file,**fields)
    frozen.write(out/'samples.json',dict(nominal=nominal,all_cases=chosen,per_arm=n))
    alpha=1-np.exp(-2*np.pi*10*.01)
    filtered=obs[:,0].clone();processed_history=obs[:,0,None].repeat(1,5)
    half=slice(2*n,3*n);low=slice(3*n,4*n)
    rows=[gpu_row(env.env).cpu().numpy()];poses=[sim.data.qpos.cpu().numpy().copy()]
    terms=[np.zeros(len(all_samples),dtype=bool)];records=[];pd_rows=[];gravity_rows=[]
    pd_error=[0.];steps=[0];wrench=[0.];contacts=[0];contact_error=[0.]
    max_nefc=[0];max_ncon=[0]
    old_apply=action.apply_actions
    def apply():
        update=action._pd_tick==0;old_apply()
        if update:
            q=sim.data.qpos[:,action._motor_q_adr];v=sim.data.qvel[:,action._motor_v_adr]
            target=action._nominal_pos+.5*action._control_actions
            legs=action._leg_action_ids;wheels=action._wheel_action_ids
            target[:,legs]=torch.maximum(torch.minimum(target[:,legs],action._reference_pos[:,legs]+3.14),action._reference_pos[:,legs]-3.14)
            target[:,wheels]=(10*action._control_actions[:,wheels]).clamp(-150.,150.)
            expected=torch.zeros_like(q);expected[:,legs]=20*(target[:,legs]-q[:,legs])-v[:,legs]
            expected[:,wheels]=.1*(target[:,wheels]-v[:,wheels]);expected=expected.clamp(-action._effort_limits,action._effort_limits)
            pd_error[0]=max(pd_error[0],float((expected-action.torques).abs().max()))
            pd_rows.append(torch.stack((target,q,v,action.torques),1).cpu().numpy())
        steps[0]+=1
    action.apply_actions=apply
    schedule=np.asarray(json.loads((out/'height-schedule.json').read_text())['time_and_height'])
    def command(s,t):
        if s.get('diagnostic'):
            k=np.searchsorted(schedule[:,0],t+1e-8,side='right')-1;k=max(0,k)
            h=float(schedule[k,1])
            if s['diagnostic']=='height_hold' and t>2.:h=.30
            return [0.,0.,h]
        cmd=s['case']['segments'][0]['command']
        for seg in s['case']['segments']:
            if t>=seg['start']-1e-9:cmd=seg['command']
        return cmd
    started=time.time()
    action.record_substeps=True
    try:
        with torch.no_grad(),patch('torch.optim.Adam.step',side_effect=AssertionError('No diagnostic learning')):
            for tick in range(3000):
                if time.time()-started>900:raise TimeoutError('Diagnostic wall limit reached')
                t=tick*.01;commands=[command(s,t) for s in chosen]*len(ARMS)
                obs=update_evaluation_commands(env,commands,obs)
                raw=obs[:,0].clone();used=raw.clone()
                if study_kind=='gyro':
                    used[half]=raw[half]*.5
                    filtered[low]=raw[low] if tick==0 else filtered[low]+alpha*(raw[low]-filtered[low])
                    used[low]=filtered[low]
                if tick==0:processed_history=used[:,None].repeat(1,5)
                else:processed_history=torch.cat((processed_history[:,1:],used[:,None]),1)
                actor_obs=obs.clone();history=env.history.clone()
                if study_kind=='gyro':
                    actor_obs[2*n:,0]=used[2*n:]
                    history.view(-1,5,25)[2*n:,:,0]=processed_history[2*n:]
                assert torch.equal(actor_obs[:,:][:,1:],obs[:,1:])
                assert torch.equal(history.view(-1,5,25)[:,:,1:],env.history.view(-1,5,25)[:,:,1:])
                assert torch.equal(actor_obs[:2*n],obs[:2*n]) and torch.equal(history[:2*n],env.history[:2*n])
                if study_kind=='projection':
                    # 复现实录图外俯仰修正，不旋转物理机身或角速度。
                    for group,degrees in ((half,-2.),(low,2.)):
                        co,si=np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees))
                        g=actor_obs[group,3:6].clone()
                        actor_obs[group,3]=co*g[:,0]+si*g[:,2]
                        actor_obs[group,5]=co*g[:,2]-si*g[:,0]
                        h=history.view(-1,5,25)[group,:,3:6].clone()
                        history.view(-1,5,25)[group,:,3]=co*h[:,:,0]+si*h[:,:,2]
                        history.view(-1,5,25)[group,:,5]=co*h[:,:,2]-si*h[:,:,0]
                    keep=[*range(3),*range(6,25)]
                    assert torch.equal(actor_obs[:,keep],obs[:,keep])
                    assert torch.equal(history.view(-1,5,25)[:,:,keep],env.history.view(-1,5,25)[:,:,keep])
                    gravity_rows.append(torch.stack((obs[:,3:6],actor_obs[:,3:6]),1).cpu().numpy())
                result,_=policy.act_inference(actor_obs,history)
                records.append(torch.stack((raw,used,history[:,-25]),-1).cpu().numpy())
                obs,_,_,_,_,_=env.step(result,auto_reset=False,resample=False)
                rows.extend(torch.stack(action.substep_rows).cpu().numpy())
                poses.append(sim.data.qpos.cpu().numpy().copy())
                terms.extend([(env.fail_ticks>100).cpu().numpy().copy()]*10)
                if tick%100==0:
                    overflow=sim.wp_data.overflow.numpy();assert not overflow.any(),overflow
                    wrench[0]=max(wrench[0],float(sim.data.xfrc_applied.abs().max()))
                    wp=sim.wp_data;nc=int(wp.nacon.numpy()[0]);max_ncon[0]=max(max_ncon[0],nc)
                    max_nefc[0]=max(max_nefc[0],int(wp.nefc.numpy().max()))
                    pairs=wp.contact.geom.numpy()[:nc]
                    take=np.isin(pairs,terrain).any(-1)&np.isin(pairs,robot).any(-1)
                    mu=wp.contact.friction.numpy()[:nc,:2][take]
                    if len(mu):
                        contacts[0]+=len(mu);contact_error[0]=max(contact_error[0],float(abs(mu-.8).max()))
                    print(json.dumps(dict(step=tick,total=3000,worlds=len(all_samples),wall_s=time.time()-started)),flush=True)
    finally:
        env.close()
    assert steps[0]==30000 and len(pd_rows)==6000
    assert pd_error[0]<2e-5 and wrench[0]==0 and contacts[0]>0 and contact_error[0]<1e-7
    assert all(torch.equal(v.cpu(),before[k]) for k,v in policy.state_dict().items())
    assert frozen.sha(out/'model_6838.pt')==SHA
    data=np.asarray(rows);qpos=np.asarray(poses);terminated=np.asarray(terms);pd=np.asarray(pd_rows);feedback=np.asarray(records)
    np.savez_compressed(out/'feedback-record.npz',time=np.arange(3000)*.01,values=feedback)
    if gravity_rows:np.savez_compressed(out/'gravity-record.npz',time=np.arange(3000)*.01,values=np.asarray(gravity_rows))
    scores={}
    for ai,arm in enumerate(ARMS):
        folder=out/arm;traces=folder/'traces';traces.mkdir(parents=True)
        fixtures=folder/'fixtures';fixtures.mkdir()
        shutil.copy2(fixture/'mapping.json',fixtures/'mapping.json')
        for i,s in enumerate(chosen):
            j=ai*n+i;end=round(s['case']['duration']/.001)+1;count=round(s['case']['duration']/.01)+1
            pose=qpos[:count,j].copy();pose[:,:3]-=env.env.scene.env_origins[j].cpu().numpy()
            tr=arrays(data[:end,j],pose,terminated=terminated[:end,j],physics_dt=.001,policy_dt=.01)
            np.savez_compressed(traces/(s['id']+'.npz'),**tr)
            shutil.copy2(fixture/(arm+'__'+s['id']+'_initial.npz'),fixtures/(s['id']+'_initial.npz'))
        np.savez_compressed(folder/'motor-pd.npz',time=np.arange(6000)*.005,values=pd[:,ai*n:(ai+1)*n])
        protocol=dict(validation_backend='mujoco_warp_gpu',mode='diagnostic_not_formal_acceptance',samples=nominal,
            intervention=arm,disturbances=conditions.record(),control=meta['control'],control_timing=meta['control_timing'])
        protocol.update(getattr(policy, 'protocol_overrides', {}).get(arm, {}))
        frozen.write(folder/'protocol.json',protocol)
        frozen.write(folder/'execution.json',dict(engine='mjlab',validation_backend='mujoco_warp_gpu',complete_traces=n))
        score=canonical(folder,'fudan_ppo_walk');frozen.write(folder/'acceptance.json',score);scores[arm]=score['technical_passed']
    frozen.write(out/'runtime-audit.json',dict(passed=True,physics_steps=steps[0],worlds=len(all_samples),
        allocated_world_physics_steps=steps[0]*len(all_samples),pd_steps=len(pd),same_state_arrays=len(paired),
        checkpoint_unchanged=True,weights_unchanged=True,optimizer_updates=0,external_wrench_peak=wrench[0],
        PD_error_nm=pd_error[0],contact_samples=contacts[0],friction_max_error=contact_error[0],
        max_nefc=max_nefc[0],max_nacon=max_ncon[0],wall_s=time.time()-started,frozen_modules=frozen.LOADED,
        nominal_success=scores,study_kind=study_kind))
    print(json.dumps(scores),flush=True)


def band(values,fs,lo=8.,hi=14.):
    f,p=periodogram(values,fs=fs,axis=0)
    take=(f>=lo)&(f<=hi)
    return np.sqrt(np.trapezoid(p[take],f[take],axis=0))


def report(out):
    from scripts.agent.motion.acceptance import compare_baseline
    samples=json.loads((out/'samples.json').read_text());chosen=samples['all_cases'];n=len(chosen)
    scores={a:json.loads((out/a/'acceptance.json').read_text()) for a in ARMS}
    comparisons={a:compare_baseline(scores[a],scores['raw']) for a in ARMS[1:]}
    arrays=0
    with np.load(out/'same-initial-state.npz') as f:
        for k in f.files:
            v=f[k];assert all(np.array_equal(v[0],x) for x in v[1:]),k;arrays+=1
    transients=[];zero=[];sham=[]
    for arm in ARMS:
        with np.load(out/arm/'motor-pd.npz') as d:
            t=d['time'];pd=d['values']
            for i,s in enumerate(chosen):
                if not s.get('diagnostic') and s['case']['category']!='zero':continue
                end=s['case']['duration'];active=(t>=8.37)&(t<13.) if s.get('diagnostic') else ((t>=end-5)&(t<end))
                with np.load(out/arm/'traces'/(s['id']+'.npz')) as tr:
                    take=(tr['time']>=8.37)&(tr['time']<13.) if s.get('diagnostic') else ((tr['time']>=end-5)&(tr['time']<end))
                    qtime=tr['qpos_time'];q=(qtime>=end-5)&(qtime<end);xy=tr['qpos'][q,:2]
                    row=dict(arm=arm,sample_id=s['id'],kind=s.get('diagnostic','zero30'),
                        target_band8to14_rms_rad=band(pd[active,i,0][:,[0,1,3,4]],200.).tolist(),
                        torque_band8to14_rms_nm=band(pd[active,i,3][:,[0,1,3,4]],200.).tolist(),
                        gyro_x_band8to14_rms_rad_s=float(band(tr['angular_body'][take,0],1000.)),
                        parking_horizontal_peak=float(np.linalg.norm(tr['linear_world'][take,:2],axis=1).max()),
                        pitch_peak_deg=float(abs(tr['roll_pitch_deg'][take,1]).max()),
                        last5_xy_path_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
                        last5_xy_net_m=float(np.linalg.norm(xy[-1]-xy[0])),
                        complete=not bool(tr['terminated'].any() or tr['numerical_failure'].any()))
                    (transients if s.get('diagnostic') else zero).append(row)
                if arm=='sham':
                    with np.load(out/'raw'/'motor-pd.npz') as raw:
                        other=raw['values'][:,i];selected=t<end
                        sham.append(dict(sample_id=s['id'],target_max_difference_rad=float(abs(pd[selected,i,0]-other[selected,0]).max())))
    frozen.write(out/'comparison.json',dict(initial_arrays_equal=arrays,nominal={a:scores[a]['technical_passed'] for a in ARMS},
        baseline_comparison=comparisons,transients=transients,zero30=zero,sham=sham,optimizer_updates=0,
        reproduction_warning='These GPU trajectories do not include invented real disturbances; compare raw amplitude with uploaded real11Hz before interpreting improvements'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    i=next(i for i,s in enumerate(chosen) if s['id']=='height_replay__50001')
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for arm in ARMS:
        with np.load(out/arm/'motor-pd.npz') as p:
            t=p['time'];select=t<15.;axes[0].plot(t[select],p['values'][select,i,0,1],label=arm,lw=.8)
            axes[1].plot(t[select],p['values'][select,i,3,1],label=arm,lw=.8)
        with np.load(out/arm/'traces/height_replay__50001.npz') as p:
            axes[2].plot(p['time'],p['angular_body'][:,0],label=arm,lw=.8)
            axes[3].plot(p['time'],np.linalg.norm(p['linear_world'][:,:2],axis=1),label=arm,lw=.8)
    for ax,label in zip(axes,('LR target (rad)','LR requested torque (Nm)','True gyroX (rad/s)','Horizontal speed (m/s)')):
        ax.set_ylabel(label);ax.axvspan(7.79,8.37,color='orange',alpha=.15);ax.grid(alpha=.2);ax.legend(ncol=4)
    axes[-1].set_xlabel('Simulated time (s)');fig.suptitle('Frozen6838 GPU: recorded height schedule / control feedback only / no real disturbance injection')
    fig.tight_layout();fig.savefig(out/'height-replay-comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(nominal={a:scores[a]['technical_passed'] for a in ARMS},
        comparison={a:c['comparison'] for a,c in comparisons.items()}),ensure_ascii=False))


def main():
    global ARMS
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','worker','report'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--study',choices=('gyro','projection'),default='gyro')
    a=p.parse_args();out=a.output.resolve()
    kind=a.study if a.mode=='run' else json.loads((out/'plan.json').read_text()).get('study_kind','gyro')
    if kind=='projection':ARMS=('raw','sham','pitch_minus2','pitch_plus2')
    if a.mode=='worker':return worker(out)
    if a.mode=='report':return report(out)
    prepare(out,kind)
    try:
        run_command(out,'gpu',[str(study.PYTHON),'-u','-m',MODULE,'worker','--output',str(out)])
        report(out);frozen.write(out/'status.json',dict(status='completed',optimizer_updates=0))
    except BaseException as exc:
        frozen.write(out/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
