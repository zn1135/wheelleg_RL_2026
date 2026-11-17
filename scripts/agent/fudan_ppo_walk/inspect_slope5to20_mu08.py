"""冻结5838/6838，同一5～20°地形和实际摩擦0.8的GPU行为对照。"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

from . import inspect_regions_stage as regional
from . import inspect_randomization_4000 as frozen
from .run_torque_ablation import run_command

ROOT=frozen.ROOT
PYTHON=ROOT/'mjlab_training/.venv/bin/python'
MODULE='scripts.agent.fudan_ppo_walk.inspect_slope5to20_mu08'
EXPERIMENT=ROOT/'artifacts/fudan_ppo_walk/20261116-scan-slope5to20-mu08-resume5838-6838-01'
RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261116_scan_slope5to20_mu08_resume5838_to6838_01'
SOURCE_RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261116_critic_scan_regions_resume5000_to6000_01'
ARMS=dict(source=(5838,SOURCE_RUN,SOURCE_RUN/'stage_5838.pt'),candidate=(6838,RUN,RUN/'model_final.pt'))


def audit(out):
    import torch
    from .resume_pd200_random_from3000 import same
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    b=json.loads((EXPERIMENT/'budget.json').read_text())
    done=json.loads((EXPERIMENT/'training-audit.json').read_text())
    assert done['budget_and_learning_state_passed'] and done['additional_iterations']==1000
    initial=torch.load(RUN/'model_initial.pt',map_location='cpu',weights_only=False)
    source=torch.load(ARMS['source'][2],map_location='cpu',weights_only=False)
    for k in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng',
              'iterations','transitions','primary_updates','encoder_updates','speed_limit','policy_steps'):
        assert same(initial[k],source[k]),k
    for arm,(iteration,run,checkpoint) in ARMS.items():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['iterations']==iteration and not saved['partial_update']
        assert saved['transitions']==iteration*4096*48+184320
        assert saved['primary_updates']==saved['encoder_updates']==iteration*20
        assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
        for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
            assert {int(v['step']) for v in saved[key]['state'].values()}=={iteration*20}
            assert all(torch.isfinite(v).all() for state in saved[key]['state'].values() for v in state.values() if isinstance(v,torch.Tensor))
        checks={};drift=[]
        for name,digest in json.loads((run/'source_sha256.json').read_text()).items():
            if name.startswith('mjlab_training/src/ct_mjlab/'):
                checks[name]=frozen.sha(run/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest
                if frozen.sha(ROOT/name)!=digest:drift.append(name)
            elif name.startswith('resources/'):checks[name]=frozen.sha(ROOT/name)==digest
        assert all(checks.values())
        folder=out/arm;folder.mkdir();(folder/'checkpoint').mkdir()
        shutil.copy2(checkpoint,folder/'checkpoint'/f'model_{iteration}.pt')
        frozen.write(folder/'checkpoint-audit.json',dict(passed=True,iteration=iteration,source=str(checkpoint),
            checkpoint_sha256=frozen.sha(checkpoint),snapshot_and_resources=checks,current_workspace_drift=drift,
            original_terrain=saved['terrain'],evaluation_terrain=b['terrain'],optimizer_updates=0))
    assert frozen.sha(ARMS['source'][2])==b['checkpoint_sha256']
    assert frozen.sha(ARMS['candidate'][2])==done['final_sha256']
    report=json.loads((RUN/'training_report.json').read_text())
    assert report['stopped_by']=='iteration_budget' and report['iterations_added']==1000
    events=EventAccumulator(str(RUN),size_guidance={'scalars':0}).Reload()
    rewards=events.Scalars('Train/reward');assert [v.step for v in rewards]==list(range(5839,6839))
    frozen.write(out/'training-audit.json',dict(passed=True,source_initial_learning_state_exact=True,
        source_iteration=5838,final_iteration=6838,additional_iterations=1000,training_wall_s=report['training_wall_s'],
        reward_first300_mean=float(np.mean([v.value for v in rewards[:300]])),
        reward_last300_mean=float(np.mean([v.value for v in rewards[-300:]])),
        speed_limit=events.Scalars('Train/speed_limit')[-1].value,
        optimizer_steps_each=136760,optimizer_updates_during_evaluation=0))


def worker(out,arm,stage):
    import torch
    iteration,run,_=ARMS[arm];root=out/arm
    frozen.RUN=regional.RUN=run;regional.EXPERIMENT=EXPERIMENT
    proof=json.loads((root/'checkpoint-audit.json').read_text())
    checkpoint=root/'checkpoint'/f'model_{iteration}.pt'
    assert proof['passed'] and frozen.sha(checkpoint)==proof['checkpoint_sha256']
    assert not any(n=='ct_mjlab' or n.startswith('ct_mjlab.') for n in sys.modules)
    sys.meta_path.insert(0,frozen.FrozenTrainingSource())
    import mujoco
    from mjlab.managers.event_manager import RecomputeLevel
    from ct_mjlab.envs.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.fudan_ppo_walk import measurement,policy as loading
    from . import evaluate
    old_configure=measurement.configure_samples;old_cfg=physical.make_env_cfg;old_load=loading.load_policy
    original_routes=regional.routes;networks=[];materials=[];contacts=[0];contact_error=[0.]
    def routes():
        samples=original_routes()
        for sample in samples:
            if sample['family'] in ('up','down'):
                k=int(sample['id'].split('_')[1]);sample['slope_deg']=5.+5.*k
        return samples
    def make_cfg(*args,**kwargs):
        cfg=old_cfg(*args,**kwargs);cfg.sim.nconmax,cfg.sim.njmax=64,256;return cfg
    def configured(wrapper,samples,folder):
        result=old_configure(wrapper,samples,folder);sim=wrapper.env.sim;m=sim.mj_model
        bodies=wrapper.robot.indexing.body_ids.cpu().numpy()
        robot=np.flatnonzero(np.isin(m.geom_bodyid,bodies))
        terrain=np.flatnonzero(m.geom_bodyid==m.body('terrain').id)
        geoms=np.union1d(robot,terrain).tolist()
        # 固定评估夹具原本恢复标称0.5；这里显式同步两侧为真实0.8。
        sim.model.geom_friction[:,geoms,0]=.8
        wrapper.env._fudan_parameters['friction'].fill_(.8)
        sim.recompute_constants(RecomputeLevel.set_const);sim.forward();wrapper.env.scene.update(dt=0.);sim.sense()
        actual=sim.model.geom_friction.cpu().numpy().copy()
        assert np.allclose(actual[:,geoms,0],.8,atol=1e-7,rtol=0.)
        for i,sample in enumerate(samples):
            path=Path(folder)/(sample['id']+'_initial.npz')
            with np.load(path) as saved:values={k:saved[k].copy() for k in saved.files}
            values['geom_friction']=actual[i];np.savez_compressed(path,**values)
        fixture=Path(folder)/'model.mjb';cpu=mujoco.MjModel.from_binary_path(str(fixture))
        cpu.geom_friction[geoms,0]=.8;mujoco.mj_saveModel(cpu,str(fixture))
        materials.append(dict(num_envs=wrapper.num_envs,robot_and_terrain_slide_friction=.8,
            terrain_geom_ids=terrain.tolist(),torsion_and_rolling_unchanged=True))
        old_step=sim.step;ticks=[0]
        def step():
            value=old_step();ticks[0]+=1
            if ticks[0]%500==0:
                data=sim.wp_data;count=int(data.nacon.numpy()[0]);pairs=data.contact.geom.numpy()[:count]
                mask=np.isin(pairs,terrain).any(-1)&np.isin(pairs,robot).any(-1)
                coeff=data.contact.friction.numpy()[:count,:2][mask]
                if len(coeff):
                    error=float(np.abs(coeff-.8).max());assert error<1e-7
                    contact_error[0]=max(contact_error[0],error);contacts[0]+=len(coeff)
            return value
        sim.step=step
        return result
    def loaded(*args,**kwargs):
        policy,meta=old_load(*args,**kwargs)
        networks.append((policy,{k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}))
        return policy,meta
    with patch.object(physical,'make_env_cfg',make_cfg),patch.object(measurement,'configure_samples',configured), \
         patch.object(evaluate,'configure_samples',configured),patch.object(loading,'load_policy',loaded), \
         patch.object(evaluate,'load_policy',loaded),patch.object(regional,'routes',routes), \
         patch('torch.optim.Adam.step',side_effect=AssertionError('Frozen comparison cannot learn')):
        if stage=='nominal':
            from . import inspect_pd200_training as inspect
            from scripts.agent.motion.acceptance import canonical
            inspect.MODEL=checkpoint;inspect.RUN=run;inspect.EXPERIMENT=EXPERIMENT
            target=root/'nominal'
            inspect.run(target,checkpoint_audit=lambda _:proof['snapshot_and_resources'],conditions_override=regional.conditions())
            score=canonical(target/'gpu','fudan_ppo_walk');(target/'acceptance').mkdir()
            frozen.write(target/'acceptance/acceptance.json',score)
            plan=json.loads((target/'plan.json').read_text());plan['is_final_checkpoint']=arm=='candidate'
            frozen.write(target/'plan.json',plan);inspect.report(target)
        elif stage=='extras':regional.extras(root,iteration)
        else:regional.terrain(root,iteration,height_sweep=stage=='height-sweep',models=[(str(iteration),checkpoint)])
    assert networks and contacts[0]>0
    assert all(torch.equal(v.cpu(),before[k]) for policy,before in networks for k,v in policy.state_dict().items())
    assert frozen.sha(checkpoint)==proof['checkpoint_sha256']
    frozen.write(root/(stage+'-frozen-execution.json'),dict(passed=True,optimizer_updates=0,
        weights_unchanged=True,checkpoint_unchanged=True,materials=materials,
        actual_contact_samples=contacts[0],maximum_contact_coefficient_error=contact_error[0],frozen_modules=frozen.LOADED))


def report(out):
    from scripts.agent.motion.acceptance import compare_baseline
    result=dict(training=json.loads((out/'training-audit.json').read_text()),nominal={},extras={},terrain={},initial_pairing={},optimizer_updates=0)
    paths={}
    for arm,(iteration,_,_) in ARMS.items():
        root=out/arm
        for label in ('nominal','high-yaw','speed3'):
            path=root/'nominal/acceptance/acceptance.json' if label=='nominal' else root/label/'acceptance.json'
            data=json.loads(path.read_text());paths.setdefault(label,{})[arm]=data
            result['nominal' if label=='nominal' else 'extras'].setdefault(label,{})[arm]=dict(passed=data['technical_passed'],total=data['total'],groups=data['groups'])
        rows=[]
        for folder in ('terrain','terrain-height-extra'):
            target=root/folder/str(iteration);data=json.loads((target/'result.json').read_text());rows+=data['rows']
            for row in data['rows']:
                with np.load(target/'traces'/(row['id']+'.npz')) as trace:
                    assert all(np.isfinite(trace[k]).all() for k in trace.files)
                    assert abs(trace['time'][-1]-trace['qpos_time'][-1])<1e-8
                    assert row['complete'] or trace['terminated'][-1]
        result['terrain'][arm]=dict(rows=rows,by_family={family:dict(total=sum(r['family']==family for r in rows),
            reached=sum(r['reached'] for r in rows if r['family']==family),passed=sum(r['passed'] for r in rows if r['family']==family),
            fell=sum(r['fell'] for r in rows if r['family']==family),incomplete=sum(not r['complete'] for r in rows if r['family']==family)) for family in ('up','down','rough')})
    for label,arms in paths.items():
        compare_baseline(arms['candidate'],arms['source'])
        result.setdefault('paired_scores',{})[label]=arms['candidate']['comparison']
    compared=0
    for folder in ('nominal/gpu','high-yaw','speed3','terrain/ITER','terrain-height-extra/ITER'):
        a=out/'source'/folder.replace('ITER','5838');b=out/'candidate'/folder.replace('ITER','6838');count=0
        for path in sorted((a/'fixtures').glob('*_initial.npz'))+([a/'initial.npz'] if (a/'initial.npz').exists() else []):
            with np.load(path) as x,np.load(b/path.relative_to(a)) as y:
                assert set(x.files)==set(y.files)
                for k in x.files:assert np.array_equal(x[k],y[k]),(folder,path.name,k)
                count+=len(x.files)
        assert count>0;compared+=count;result['initial_pairing'][folder]=count
    for protocol in ('terrain-protocol.json','terrain-height-extra-protocol.json'):
        assert json.loads((out/'source'/protocol).read_text())==json.loads((out/'candidate'/protocol).read_text())
    result['initial_arrays_bitwise_equal']=compared
    frozen.write(out/'comparison.json',result)
    return result


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(RUN),size_guidance={'scalars':0}).Reload()
    values=events.Scalars('Train/reward');x=np.array([v.step for v in values]);y=np.array([v.value for v in values])
    fig,ax=plt.subplots(figsize=(10,3));ax.plot(x,y,lw=.5,alpha=.25)
    ax.plot(x[49:],np.convolve(y,np.ones(50)/50,mode='valid'));ax.set(xlabel='Iteration',ylabel='Reward',title='5838 -> 6838 / new recipe only')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'training-reward.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True)
    for row,family in enumerate(('up','down')):
        for k in range(4):
            sid=f'pyramid_{k}_{family}_v1';ax=axes[row,k]
            for arm,(iteration,_,_) in ARMS.items():
                base=out/arm/'terrain'/str(iteration)
                case=next(r for r in json.loads((base/'result.json').read_text())['rows'] if r['id']==sid)
                with np.load(base/'traces'/(sid+'.npz')) as d:
                    progress=(d['qpos'][:,0]-case['start_xy'][0])*(1 if family=='up' else -1)
                    ax.plot(d['qpos_time'],progress,label=str(iteration))
            ax.axhline(4.,color='k',ls='--',lw=.8);ax.axvline(4.,color='gray',ls=':',lw=.8)
            ax.set(title=f'{family} / {5+5*k}deg',xlabel='Time (s)',ylabel='World x progress (m)');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'terrain-progress.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex='col');focus={}
    for col,speed in enumerate((.5,1.)):
        sid=f'pyramid_3_up_v{speed:g}'
        for arm,(iteration,_,_) in ARMS.items():
            base=out/arm/'terrain'/str(iteration)
            case=next(r for r in json.loads((base/'result.json').read_text())['rows'] if r['id']==sid)
            with np.load(base/'traces'/(sid+'.npz')) as d:
                t=d['time'];mask=(t>=2)&(t<case['drive_s'])
                # 50ms分箱用于显示；统计保持完整1ms记录。
                length=(len(t)-1)//50*50;tx=t[:length].reshape(-1,50).mean(-1)
                for ax,key in zip(axes[:,col],('linear_body','base_contact','roll_pitch_deg')):
                    y=d[key][:length]
                    if key=='linear_body':y=y[:,0]
                    if key=='roll_pitch_deg':y=y[:,1]
                    ax.plot(tx,y.reshape(-1,50).mean(-1),label=str(iteration));ax.grid(alpha=.25);ax.legend()
                focus[f'{arm}_v{speed:g}']=dict(mean_vx_m_s=float(d['linear_body'][mask,0].mean()),
                    drive_body_contact_mean_N=float(d['base_contact'][mask].mean()),
                    drive_body_contact_peak_N=float(d['base_contact'][mask].max()),
                    drive_body_contact_over10_fraction=float((d['base_contact'][mask]>10).mean()),
                    complete=case['complete'],reached=case['reached'],endpoint_error_m=case['endpoint_error_m'])
        for ax,label in zip(axes[:,col],('Actual vx (m/s)','Body contact (N)','Pitch (deg)')):
            ax.set_ylabel(label);ax.axvline(4./speed,color='gray',ls=':',lw=.8)
        axes[0,col].set_title(f'20deg uphill / command={speed:g}m/s / h=.26m')
        axes[-1,col].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(out/'20deg-speed-contact.png',dpi=150);plt.close(fig)
    frozen.write(out/'20deg-details.json',focus)


def render(out):
    view=out/'video-inputs';view.mkdir(exist_ok=False)
    for folder in ('terrain','terrain-height-extra'):
        (view/folder).mkdir()
        for arm,(iteration,_,_) in ARMS.items():
            (view/folder/str(iteration)).symlink_to((out/arm/folder/str(iteration)).resolve(),target_is_directory=True)
    regional.render(view,6838,labels=('5838','6838'),
        chosen=['pyramid_1_up_v1','pyramid_3_up_v1','pyramid_3_down_v0.5_h0.3','rough_0_dir1_v1'],output_dir=out/'videos')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','worker','report','render','plot'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--stage',choices=('nominal','extras','terrain','height-sweep'))
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='worker':return worker(out,a.arm,a.stage)
    if a.mode in ('report','render','plot'):return globals()[a.mode](out)
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'inspection-source.py');audit(out)
    try:
        for stage in ('terrain','height-sweep','nominal','extras'):
            for arm in ARMS:
                run_command(out,f'{arm}-{stage}',[str(PYTHON),'-u','-m',MODULE,'worker','--output',str(out),'--arm',arm,'--stage',stage])
        report(out)
        frozen.write(out/'status.json',dict(status='completed',optimizer_updates=0,training_started=False,finished_unix=time.time()))
    except BaseException as exc:
        frozen.write(out/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
