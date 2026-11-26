"""冻结6838：10°停车状态分支，原估速/同态估速对照/同态真实速度。"""
import argparse
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import numpy as np
from . import inspect_slope5to20_mu08 as study
from . import inspect_regions_stage as regional
from . import inspect_randomization_4000 as frozen
from .run_torque_ablation import run_command

ROOT=frozen.ROOT
MODULE='scripts.agent.fudan_ppo_walk.diagnose_ten_degree_parking'
SHA='8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad'
ARMS=('estimated','sham','truth','vx_truth')


def routes():
    cases=[]
    for h in (.22,.26,.30):
        for v in (.5,1.):
            sid=f'pyramid_1_up_v{v:g}'+(f'_h{h:g}' if h!=.26 else '')
            cases.append(dict(pair_id=sid,family='up',slope_deg=10.,height=h,command_vx=v,
                start_xy=(8.,4.),target_xy=(12.,4.),drive_s=4./v,duration=4./v+3.))
    return [dict(s,id=arm+'__'+s['pair_id'],arm=arm) for arm in ARMS for s in cases]


def prepare(out):
    source=study.RUN/'model_final.pt';assert frozen.sha(source)==SHA
    checks={}
    for name,digest in json.loads((study.RUN/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            checks[name]=frozen.sha(study.RUN/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest
        elif name.startswith('resources/'):checks[name]=frozen.sha(ROOT/name)==digest
    assert all(checks.values())
    out.mkdir(parents=True,exist_ok=False);(out/'checkpoint').mkdir()
    shutil.copy2(source,out/'checkpoint/model_6838.pt');shutil.copy2(__file__,out/'diagnostic-source.py')
    frozen.write(out/'checkpoint-audit.json',dict(passed=True,checkpoint_sha256=SHA,snapshot_and_resources=checks))
    frozen.write(out/'plan.json',dict(samples=routes(),model_iteration=6838,friction=.8,same_state_fork=True,
        intervention='At zero-command boundary copy estimated state to all branches; truth replaces xyz velocity, vx_truth replaces vx only',
        truth='Current training supervision: body-frame last10ms position difference, scaled by2',
        sham='Identical state copy and original encoder; measures numerical divergence after branching',
        timing='Physics1ms / PD5ms / policy10ms / original dual10-20ms delay queues',
        window='drive_s+1s through drive_s+3s; only post-fork parking is assessed',
        optimizer_updates=0,training_started=False))


def worker(out):
    import torch
    frozen.RUN=regional.RUN=study.RUN;regional.EXPERIMENT=study.EXPERIMENT
    checkpoint=out/'checkpoint/model_6838.pt';assert frozen.sha(checkpoint)==SHA
    assert not any(n=='ct_mjlab' or n.startswith('ct_mjlab.') for n in sys.modules)
    sys.meta_path.insert(0,frozen.FrozenTrainingSource())
    import mujoco
    from mjlab.managers.event_manager import RecomputeLevel
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.fudan_ppo_walk import measurement,policy as loading
    old_configure=measurement.configure_samples;old_cfg=physical.make_env_cfg
    old_init=physical.FudanEnv.__init__;old_load=loading.load_policy;old_write=frozen.write
    samples=routes();envs=[];networks=[];records=[];switches={};ticks=[0];contacts=[0]
    def configure(e,batch,folder):
        shared=[dict(s,case=dict(s['case'],id=s['id'].split('__',1)[1])) for s in batch]
        result=old_configure(e,shared,folder);sim=e.env.sim;m=sim.mj_model
        bodies=e.robot.indexing.body_ids.cpu().numpy()
        terrain=np.flatnonzero(m.geom_bodyid==m.body('terrain').id)
        robot=np.flatnonzero(np.isin(m.geom_bodyid,bodies));geoms=np.union1d(terrain,robot).tolist()
        sim.model.geom_friction[:,geoms,0]=.8;e.env._fudan_parameters['friction'].fill_(.8)
        sim.recompute_constants(RecomputeLevel.set_const);sim.forward();e.env.scene.update(dt=0.);sim.sense()
        values=sim.model.geom_friction.cpu().numpy().copy();assert np.allclose(values[:,geoms,0],.8,atol=1e-7,rtol=0)
        for i,s in enumerate(samples):
            path=Path(folder)/(s['id']+'_initial.npz')
            with np.load(path) as saved:fields={k:saved[k].copy() for k in saved.files}
            fields['geom_friction']=values[i];np.savez_compressed(path,**fields)
        fixture=Path(folder)/'model.mjb';cpu=mujoco.MjModel.from_binary_path(str(fixture))
        cpu.geom_friction[geoms,0]=.8;mujoco.mj_saveModel(cpu,str(fixture))
        old_step=sim.step;count=[0]
        def step():
            value=old_step();count[0]+=1
            if count[0]%500==0:
                data=sim.wp_data;n=int(data.nacon.numpy()[0]);pairs=data.contact.geom.numpy()[:n]
                mask=np.isin(pairs,terrain).any(-1)&np.isin(pairs,robot).any(-1)
                coeff=data.contact.friction.numpy()[:n,:2][mask]
                if len(coeff):assert np.max(np.abs(coeff-.8))<1e-7
                contacts[0]+=len(coeff)
            return value
        sim.step=step
        return result
    def cfg(*args,**kwargs):
        c=old_cfg(*args,**kwargs);c.sim.nconmax,c.sim.njmax=64,256;return c
    def initialized(e,*args,**kwargs):old_init(e,*args,**kwargs);envs.append(e)
    def fork(e,obs,history,i):
        sim=e.env.sim;dest=[i+6*k for k in range(1,len(ARMS))];saved={}
        # MuJoCo积分输入、求解器热启动、闭链开关及施力全部分支。
        for name in ('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','time','eq_active'):
            value=getattr(sim.data,name);value[dest]=value[i].clone();saved['physics.'+name]=value
        for label,owner in (('wrapper',e),('pd',e.action),('delay',e.action.disturbances),('motor',e.action.wheel_motor)):
            for name,value in vars(owner).items():
                if not isinstance(value,torch.Tensor) or name in ('ids','_env_ids','env_ids'):continue
                if name in ('_action_history','actions','sensors') and value.ndim>1 and value.shape[1]==len(samples):
                    value[:,dest]=value[:,i:i+1].clone();saved[label+'.'+name]=value.transpose(0,1)
                elif value.ndim and value.shape[0]==len(samples):
                    value[dest]=value[i].clone();saved[label+'.'+name]=value
        obs[dest]=obs[i].clone();history[dest]=history[i].clone()
        saved['actor.obs']=obs;saved['actor.history']=history
        sim.forward();e.env.scene.update(dt=0.);sim.sense()
        assert torch.equal(e.action.disturbances.ids,torch.arange(len(samples),device=e.device))
        result={k:v[[i,*dest]].detach().cpu().numpy().copy() for k,v in saved.items()}
        for k,v in result.items():assert all(np.array_equal(v[0],other) for other in v[1:]),k
        return result
    def loaded(*args,**kwargs):
        policy,meta=old_load(*args,**kwargs);original=policy.act_inference;current={}
        networks.append((policy,{k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}))
        def actor_input(module,args):
            e=envs[0];x=args[0];assert x.shape==(len(samples),28) and e.cfg.normalization.command[0]==2.
            truth=e.linear*2;mask=current['mask'];used=x.clone();used[mask,25:]=truth[mask]
            vx_mask=current['vx_mask'];used[vx_mask,25]=truth[vx_mask,0]
            assert torch.equal(used[:,:25],x[:,:25]) and torch.equal(used[~(mask|vx_mask)],x[~(mask|vx_mask)])
            assert torch.equal(used[vx_mask,26:],x[vx_mask,26:])
            current.update(estimated=x[:,25:].clone()/2,reference=truth.clone()/2,used=used[:,25:].clone()/2)
            return (used,)
        policy.actor.register_forward_pre_hook(actor_input)
        def inference(obs,history):
            e=envs[0];t=ticks[0]*e.dt
            for i,s in enumerate(samples[:6]):
                if ticks[0]==round(s['drive_s']/e.dt):switches[s['pair_id']]=fork(e,obs,history,i)
            current['mask']=torch.tensor([s['arm']=='truth' and s['drive_s']<=t<s['duration'] for s in samples],device=e.device)
            current['vx_mask']=torch.tensor([s['arm']=='vx_truth' and s['drive_s']<=t<s['duration'] for s in samples],device=e.device)
            qpos=e.env.sim.data.qpos.clone();qvel=e.env.sim.data.qvel.clone();result=original(obs,history)
            records.append(dict(time_s=t,qpos=qpos.cpu().numpy(),qvel=qvel.cpu().numpy(),
                estimated=current['estimated'].cpu().numpy(),reference=current['reference'].cpu().numpy(),
                actor_velocity=current['used'].cpu().numpy(),action=result[0].cpu().numpy(),intervention=(current['mask']|current['vx_mask']).cpu().numpy()))
            ticks[0]+=1;return result
        policy.act_inference=inference
        return policy,meta
    def write(path,value):
        if Path(path).name=='terrain-protocol.json':
            value=dict(value,version='ten-degree-parking-state-branch-v2',heights_m=[.22,.26,.30],same_state_fork=True,
                intervention=json.loads((out/'plan.json').read_text())['intervention'],parking_only_diagnostic=True)
        return old_write(path,value)
    destination=out/'gpu';destination.mkdir()
    with patch.object(physical,'make_env_cfg',cfg),patch.object(physical.FudanEnv,'__init__',initialized),patch.object(measurement,'configure_samples',configure),patch.object(loading,'load_policy',loaded),patch.object(regional,'routes',routes),patch.object(frozen,'write',write),patch('torch.optim.Adam.step',side_effect=AssertionError('No diagnostic learning')):
        regional.terrain(destination,6838,models=[('6838',checkpoint)])
    assert ticks[0]==1100 and len(switches)==6 and contacts[0]>0
    assert all(torch.equal(v.cpu(),before[k]) for policy,before in networks for k,v in policy.state_dict().items())
    np.savez_compressed(out/'policy-record.npz',**{k:np.asarray([r[k] for r in records]) for k in records[0]})
    folder=out/'switch-state';folder.mkdir()
    for sid,data in switches.items():np.savez_compressed(folder/(sid+'.npz'),**data)
    frozen.write(out/'runtime-audit.json',dict(passed=True,weights_unchanged=True,checkpoint_unchanged=frozen.sha(checkpoint)==SHA,
        optimizer_updates=0,num_envs=len(samples),policy_steps=1100,physics_steps=11000,actual_contact_samples=contacts[0],actual_contact_friction_error=0.,frozen_modules=frozen.LOADED))


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    samples=routes();root=out/'gpu/terrain/6838'
    by_id={r['id']:r for r in json.loads((root/'result.json').read_text())['rows']}
    checked=0
    for s in samples[:6]:
        with np.load(out/'switch-state'/(s['pair_id']+'.npz')) as d:
            for k in d.files:
                v=d[k];assert all(np.array_equal(v[0],other) for other in v[1:]),(s['pair_id'],k)
                checked+=1
    stats=[]
    with np.load(out/'policy-record.npz') as p:
        for i,s in enumerate(samples):
            row=by_id[s['id']]
            with np.load(root/'traces'/(s['id']+'.npz')) as d:
                assert all(np.isfinite(d[k]).all() for k in d.files)
                end=float(d['time'][-1]);mask=(d['time']>=s['drive_s']+1)&(d['time']<end)
                valid=(p['time_s']>=s['drive_s']+1)&(p['time_s']<end)
                assert mask.any() and valid.any()
                qs=(d['qpos_time']>=s['drive_s']+1)&(d['qpos_time']<end);xy=d['qpos'][qs,:2]
                selected=p['intervention'][:,i];assert not selected[p['time_s']<s['drive_s']].any()
                if selected.any():
                    if s['arm']=='vx_truth':
                        assert np.array_equal(p['actor_velocity'][selected,i,0],p['reference'][selected,i,0])
                        assert np.array_equal(p['actor_velocity'][selected,i,1:],p['estimated'][selected,i,1:])
                    else:assert np.array_equal(p['actor_velocity'][selected,i],p['reference'][selected,i])
                err=p['estimated'][valid,i]-p['reference'][valid,i]
                stats.append(dict(sample_id=s['id'],pair_id=s['pair_id'],arm=s['arm'],height_m=s['height'],speed_command=s['command_vx'],complete=row['complete'],
                    parking_passed=row['stop']['passed'],velocity_peak_passed=row['stop']['checks']['stopped'],
                    parking_horizontal_peak_m_s=float(np.linalg.norm(d['linear_world'][mask,:2],axis=1).max()),
                    parking_mean_vx_m_s=float(d['linear_body'][mask,0].mean()),parking_xy_path_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
                    parking_xy_net_m=float(np.linalg.norm(xy[-1]-xy[0])),parking_pitch_peak_deg=float(np.abs(d['roll_pitch_deg'][mask,1]).max()),
                    encoder_bias_vx_m_s=float(err[:,0].mean()),encoder_MAE_vx_m_s=float(np.abs(err[:,0]).mean()),both_wheel_support=row['both_wheel_20ms_support_ok']))
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True);sid='pyramid_1_up_v1'
    with np.load(out/'policy-record.npz') as p:
        for arm in ARMS:
            key=arm+'__'+sid;i=next(i for i,s in enumerate(samples) if s['id']==key)
            with np.load(root/'traces'/(key+'.npz')) as d:
                start=0 if arm=='estimated' else 4.;active=(d['time']>=start)&(d['time']<7)
                axes[0].plot(d['time'][active],d['linear_body'][active,0],label=arm,lw=.8)
                axes[2].plot(d['time'][active],d['roll_pitch_deg'][active,1],label=arm,lw=.8)
            active=(p['time_s']>=start)&(p['time_s']<7);axes[1].plot(p['time_s'][active],p['actor_velocity'][active,i,0],label=arm,lw=.8)
    for ax,title in zip(axes,('Actual vx (m/s)','Actor velocity feedback (m/s)','Pitch (deg)')):
        ax.axvline(4,color='k',ls='--');ax.axvspan(5,7,color='grey',alpha=.15);ax.set_ylabel(title);ax.grid(alpha=.25);ax.legend()
    axes[-1].set_xlabel('Time (s)');fig.suptitle('Frozen6838 / 10deg uphill / h=.26m / same-state parking fork / friction=.8')
    fig.tight_layout();fig.savefig(out/'parking-comparison.png',dpi=150);plt.close(fig)
    result=dict(passed=True,same_state_fork_arrays_bitwise_equal=checked,takeover_state_max_error=0.,rows=stats,optimizer_updates=0,training_started=False)
    frozen.write(out/'comparison.json',result);print(json.dumps(result,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','worker','report'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='worker':return worker(out)
    if a.mode=='report':return report(out)
    prepare(out)
    try:
        run_command(out,'gpu',[str(study.PYTHON),'-u','-m',MODULE,'worker','--output',str(out)])
        report(out);frozen.write(out/'status.json',dict(status='completed',optimizer_updates=0))
    except BaseException as exc:
        frozen.write(out/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
