"""Audit the completed randomization continuation and evaluate its frozen training source."""
import argparse
from dataclasses import replace
import difflib
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261115_pd200_full_random_force3_push_resume3018_to4000_01'
EXPERIMENT=ROOT/'artifacts/fudan_ppo_walk/20261115-pd200-resume3018-4000-01'
BASELINE=ROOT/'artifacts/fudan_ppo_walk/20261114-pd200-delay-resume1000-3000-01/final-3000-01'
LOADED={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


class FrozenTrainingSource(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='ct_mjlab' and not fullname.startswith('ct_mjlab.'):return None
        relative=Path(*fullname.split('.'));saved=RUN/'source'/relative
        package=saved.is_dir();saved=saved/'__init__.py' if package else saved.with_suffix('.py')
        if not saved.exists():raise ImportError(f'Module absent from frozen training source: {fullname}')
        original=ROOT/'mjlab_training/src'/relative
        original=original/'__init__.py' if package else original.with_suffix('.py')
        class Loader(importlib.abc.Loader):
            def create_module(self,spec):return None
            def exec_module(self,module):
                # 保留原__file__用于定位机器人资源，执行内容严格取已校验训练快照。
                LOADED[fullname]=dict(snapshot=str(saved),sha256=sha(saved),resource_origin=str(original))
                exec(compile(saved.read_bytes(),str(original),'exec'),module.__dict__)
        return importlib.util.spec_from_file_location(fullname,original,loader=Loader(),
            submodule_search_locations=[str(original.parent)] if package else None)


def audit(out):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from scripts.agent.fudan_ppo_walk.resume_pd200_random_from3000 import same
    out.mkdir(parents=True,exist_ok=False)
    report=json.loads((RUN/'training_report.json').read_text())
    budget=json.loads((EXPERIMENT/'budget.json').read_text())
    source=torch.load(budget['checkpoint'],map_location='cpu',weights_only=False)
    initial=torch.load(RUN/'model_initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(RUN/'model_final.pt',map_location='cpu',weights_only=False)
    assert sha(RUN/'model_final.pt')==report['checkpoint_sha256']
    assert sha(budget['checkpoint'])==budget['checkpoint_sha256']
    assert json.loads((EXPERIMENT/'process-result.json').read_text())['returncode']==0
    assert json.loads((EXPERIMENT/'initialization-audit.json').read_text())['passed']
    for key in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng',
                'transitions','primary_updates','encoder_updates','speed_limit','policy_steps'):
        assert same(initial[key],source[key]),key
    assert source['iterations']==initial['iterations']==3018 and final['iterations']==4000 and not final['partial_update']
    assert report['stopped_by']=='iteration_budget' and report['iterations_added']==report['iteration_budget']==982
    assert final['transitions']-source['transitions']==982*4096*48
    assert final['transitions']==4000*4096*48+184320
    assert final['primary_updates']==final['encoder_updates']==80000
    for key in ('control','control_timing','height_reward','disturbances','physical_wheel_model'):
        assert initial[key]==final[key]==source[key],key
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in final[key]['state'].values()}=={80000}
        assert all(torch.isfinite(v).all() for state in final[key]['state'].values()
                   for v in state.values() if isinstance(v,torch.Tensor))
    assert all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
    events=EventAccumulator(str(RUN),size_guidance={'scalars':0}).Reload()
    for tag in ['Train/reward',*['Rewards/'+name for name in final['reward_names']]]:
        assert [v.step for v in events.Scalars(tag)]==list(range(3019,4001)),tag
    manifest=json.loads((RUN/'source_sha256.json').read_text());checks={};drift=[];diff=[]
    for name,digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            saved=RUN/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/')
            checks[name]=sha(saved)==digest;assert checks[name],name
            if sha(ROOT/name)!=digest:
                drift.append(name);diff.extend(difflib.unified_diff(saved.read_text().splitlines(True),
                    (ROOT/name).read_text().splitlines(True),fromfile='training/'+name,tofile='workspace/'+name))
        elif name.startswith('resources/'):
            checks[name]=sha(ROOT/name)==digest;assert checks[name],name
    (out/'workspace-drift.patch').write_text(''.join(diff))
    (out/'checkpoint').mkdir();shutil.copy2(RUN/'model_final.pt',out/'checkpoint/model_final.pt')
    result=dict(passed=True,training_completed=True,checkpoint_sha256=sha(RUN/'model_final.pt'),
        source_iteration=3018,final_iteration=4000,additional_iterations=982,original_authorized_iterations=1000,
        completed_iterations_before_resume=18,completed_update_transitions=4000*4096*48,
        recorded_cumulative_transitions=final['transitions'],inherited_abandoned_rollout_transitions=184320,
        optimizer_updates_added_each=19640,training_wall_s=report['training_wall_s'],
        snapshot_and_resource_checks=checks,current_workspace_drift=drift,
        original_source_gate_passed=False,original_source_gate_error=json.loads((EXPERIMENT/'status.json').read_text()),
        evaluation_execution='Verified frozen ct_mjlab source; original paths retained only for resource resolution',
        optimizer_updates_during_inspection=0,behavior_accepted=False)
    write(out/'checkpoint-audit.json',result)
    shutil.copy2(__file__,out/'inspection-source.py')
    print(json.dumps({k:v for k,v in result.items() if k!='snapshot_and_resource_checks'},ensure_ascii=False))


def stress(out):
    import numpy as np
    import torch
    from scripts.agent.fudan_ppo_walk import inspect_pd200_training as inspect, evaluate
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import from_record
    from mjlab.managers.event_manager import RecomputeLevel
    torch.set_num_threads(1)
    chosen=[sample for sample in inspect.items() if sample['case']['category'] in ('zero','stop_forward','stop_backward')]
    cases=[dict(sample,id=f'mu{mu:g}__'+sample['id'],friction_coefficient=mu)
           for mu in (.4,.8,1.2) for sample in chosen]
    assert len(cases)==27
    final=out/'checkpoint/model_final.pt';conditions=from_record(torch.load(final,map_location='cpu',weights_only=False)['disturbances'])
    original=evaluate.configure_samples
    for label,model in [('baseline3000',Path(json.loads((BASELINE/'plan.json').read_text())['checkpoint'])),('candidate4000',final)]:
        record=[];contacts=[0];maximum_error=[0.];push_count=[0];force_peak=[0.]
        def configure(wrapper,batch,folder):
            obs,critic=original(wrapper,batch,folder);env=wrapper.env;sim=env.sim;action=wrapper.action
            bodies=set(wrapper.robot.indexing.body_ids.cpu().tolist())
            geoms=[i for i,body in enumerate(sim.mj_model.geom_bodyid) if int(body) in bodies]
            terrain=sim.mj_model.geom('terrain').id;coefficient=torch.tensor([s['friction_coefficient'] for s in batch],device=wrapper.device)
            sim.model.geom_friction[:,geoms,0]=coefficient[:,None];sim.model.geom_friction[:,terrain,0]=coefficient
            env._fudan_parameters['friction'].copy_(coefficient)
            sim.recompute_constants(RecomputeLevel.set_const);sim.forward();env.scene.update(dt=0.);sim.sense()
            # 夹具同步记录实际材料，供两模型逐数组配对。
            for i,sample in enumerate(batch):
                path=Path(folder)/(sample['id']+'_initial.npz')
                with np.load(path) as saved:values={key:saved[key].copy() for key in saved.files}
                values['geom_friction']=sim.model.geom_friction[i].cpu().numpy().copy();np.savez_compressed(path,**values)
            state=action.disturbances;force=state.force;apply=action.apply_actions;ticks=[0]
            def observed_force(quat,scale):
                value=force(quat,scale);push_count[0]+=int(value.ne(0).any(-1).sum());return value
            state.force=observed_force
            def observed_apply():
                apply();ticks[0]+=1;force_peak[0]=max(force_peak[0],float(sim.data.xfrc_applied.abs().max()))
                if ticks[0]%1000==0:
                    data=sim.wp_data;count=int(data.nacon.numpy()[0]);pairs=data.contact.geom.numpy()[:count]
                    mask=(pairs==terrain).any(-1);world=data.contact.worldid.numpy()[:count][mask]
                    actual=data.contact.friction.numpy()[:count,:2][mask];expected=coefficient.cpu().numpy()[world,None]
                    if len(actual):
                        error=float(np.max(np.abs(actual-expected)));assert error<1e-7
                        maximum_error[0]=max(maximum_error[0],error);contacts[0]+=len(actual)
            action.apply_actions=observed_apply
            record.append(dict(num_envs=wrapper.num_envs,friction_coefficient=coefficient.cpu().tolist(),
                episode_wrench_body=state.episode_wrench.cpu().tolist(),action_steps=state.action_steps.cpu().tolist(),
                sensor_steps=state.sensor_steps.cpu().tolist(),push_interval_s=7.,max_push_velocity_scale_m_s=.3))
            return obs,critic
        target=out/'stress'/label
        args=SimpleNamespace(checkpoint=model,output=target,deadline=time.time()+900,diagnostic=True,
            diagnostic_height=None,batch_size=27,selection=None,leg_load_profile='legacy',response_parameters=None)
        with patch.object(evaluate,'evaluation_items',return_value=cases),patch.object(evaluate,'from_sim2sim_config',return_value=conditions),patch.object(evaluate,'configure_samples',configure):
            evaluate.gpu(args)
        assert contacts[0]>0 and push_count[0]>0
        write(target/'stress-runtime-audit.json',dict(passed=True,actual_parameters=record,
            sampled_ground_contacts=contacts[0],contact_friction_max_error=maximum_error[0],
            push_event_observations=push_count[0],world_force_peak_N=force_peak[0],optimizer_updates=0,
            scope='27 paired parking/zero cases at3 fixed friction values with3N body-axis force and7s single-tick pushes; not full randomization acceptance'))


def compare(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    current=out/'nominal4000';old=json.loads((BASELINE/'summary.json').read_text());new=json.loads((current/'summary.json').read_text())
    old_rows={r['sample_id']:r for r in old['rows']};new_rows={r['sample_id']:r for r in new['rows']}
    paired_arrays=0
    for sid in old_rows:
        with np.load(BASELINE/'gpu/fixtures'/(sid+'_initial.npz')) as before,np.load(current/'gpu/fixtures'/(sid+'_initial.npz')) as after:
            assert set(before.files)==set(after.files)
            for field in before.files:assert np.array_equal(before[field],after[field]),(sid,field)
            paired_arrays+=len(before.files)
    plan=json.loads((current/'plan.json').read_text());motors=[np.load(directory/'motor-pd200.npz') for directory in (BASELINE,current)]
    zero=[];fig,axes=plt.subplots(3,2,figsize=(12,8))
    for row,height in enumerate((.22,.26,.30)):
        idx=next(i for i,s in enumerate(plan['samples']) if s['case']['category']=='zero' and s['case']['segments'][0]['command'][2]==height)
        sid=plan['samples'][idx]['id'];values=[]
        for label,directory,motor,records in zip(('3000','4000'),(BASELINE,current),motors,(old_rows,new_rows)):
            t=motor['time'];data=motor['values'];last=(t>=29)&(t<30)
            leg_pp=float(np.ptp(np.rad2deg(data[last,idx][:,[6,7,9,10]]),axis=0).max())
            target_pp=float(np.ptp(np.rad2deg(data[last,idx][:,[0,1,3,4]]),axis=0).max())
            values.append(dict(model=label,actual_leg_last1_pp_deg_200Hz=leg_pp,target_leg_last1_pp_deg_200Hz=target_pp,
                total_displacement_m=records[sid]['total_displacement_m'],passed=records[sid]['passed']))
            select=(t>=25)&(t<30);axes[row,0].plot(t[select],np.rad2deg(data[select,idx,7]),label=label,lw=1.)
            with np.load(directory/'gpu/traces'/(sid+'.npz')) as trace:
                displacement=np.linalg.norm(trace['qpos'][:,:2]-trace['qpos'][0,:2],axis=1)
                axes[row,1].plot(trace['qpos_time'],displacement,label=label,lw=1.)
        zero.append(dict(height_m=height,sample_id=sid,models=values))
        axes[row,0].set(title=f'Zero command h={height:.2f}m / left rear actual',xlabel='Time (s)',ylabel='Joint angle (deg)')
        axes[row,1].set(title=f'Zero command h={height:.2f}m / displacement',xlabel='Time (s)',ylabel='Distance from start (m)')
        for ax in axes[row]:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'comparison-zero.png',dpi=150);plt.close(fig)
    accepted=json.loads((current/'acceptance/acceptance.json').read_text())
    result=dict(nominal_before_passed=old['technical_passed'],nominal_after_passed=new['technical_passed'],nominal_total=47,
        paired_initial_arrays=paired_arrays,comparison=accepted['comparison'],zero_command=zero,reward=new['reward'],
        group_counts_before=old['groups'],group_counts_after=new['groups'],optimizer_updates=0,behavior_accepted=False)
    stress_results={}
    for label in ('baseline3000','candidate4000'):
        path=out/'stress'/label/'acceptance/acceptance.json'
        if not path.exists():continue
        acceptance=json.loads(path.read_text());rows=[]
        protocol=json.loads((path.parents[1]/'protocol.json').read_text())
        for sample,score in zip(protocol['samples'],acceptance['samples']):
            with np.load(path.parents[1]/'traces'/(sample['id']+'.npz')) as trace:
                end=sample['case']['duration'];last=(trace['time']>=end-5)&(trace['time']<end)
                rows.append(dict(sample_id=sample['id'],category=sample['case']['category'],friction=sample['friction_coefficient'],
                    passed=score['passed'],complete=score['complete'],finite=score['finite'],
                    failure_reasons=score['failure_reasons'],last5_speed_peak_m_s=float(np.linalg.norm(trace['linear_world'][last,:2],axis=1).max()),
                    last5_body_contact_peak_N=float(trace['base_contact'][last].max()),
                    last5_pitch_peak_deg=float(abs(trace['roll_pitch_deg'][last,1]).max()),
                    last5_wheel_support_mean_N=trace['wheel_support'][last].mean(0).tolist()))
        stress_results[label]=dict(passed=acceptance['technical_passed'],total=acceptance['total'],rows=rows)
    if stress_results:result['friction_force_push_stress']=stress_results
    write(out/'comparison.json',result)
    print(json.dumps({key:result[key] for key in ('nominal_before_passed','nominal_after_passed','paired_initial_arrays','zero_command')},ensure_ascii=False))


def videos(out):
    import imageio.v2 as imageio
    checked=[]
    for path in sorted((out/'paired-videos').glob('*.mp4')):
        metadata=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==metadata['video_sha256']
        for column in metadata['columns']:
            assert sha(column['checkpoint'])==column['model_sha256']
            assert sha(column['trajectory'])==column['trajectory_sha256']
        reader=imageio.get_reader(path);fps=reader.get_meta_data()['fps'];count=0;still=None
        for frame in reader:
            if count==round((metadata['duration_s']-1)*fps):still=frame.copy()
            count+=1
        reader.close();assert fps==50 and count==round(metadata['duration_s']*50) and still is not None
        imageio.imwrite(path.with_suffix('.png'),still)
        checked.append(dict(file=str(path),frames=count,fps=fps,playback_speed=1,full_decode_passed=True,
            video_sha256=sha(path),input_trajectories=metadata['columns']))
    assert len(checked)==3
    write(out/'paired-videos/video-index.json',dict(passed=True,videos=checked,physics_steps=0,optimizer_updates=0))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('audit','run','report','render','stress','compare','videos'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
    if args.mode=='audit':audit(out);return
    if args.mode=='compare':compare(out);return
    if args.mode=='videos':videos(out);return
    proof=json.loads((out/'checkpoint-audit.json').read_text());assert proof['passed']
    model=out/'checkpoint/model_final.pt';assert sha(model)==proof['checkpoint_sha256']
    sys.meta_path.insert(0,FrozenTrainingSource())
    if args.mode=='stress':
        stress(out);write(out/'stress-frozen-execution.json',dict(modules=LOADED,optimizer_updates=0));return
    from scripts.agent.fudan_ppo_walk import inspect_pd200_training as inspect
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import from_record
    import torch
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    inspect.MODEL=model;inspect.RUN=RUN;inspect.EXPERIMENT=EXPERIMENT
    clean=replace(from_record(metadata['disturbances']),pushes=False,external_force_range=(0.,0.),external_torque_range=(0.,0.))
    target=out/'nominal4000'
    if args.mode=='run':
        inspect.run(target,checkpoint_audit=lambda _:proof['snapshot_and_resource_checks'],conditions_override=clean)
        plan=json.loads((target/'plan.json').read_text());plan.update(training_disturbances=metadata['disturbances'],
            evaluation_disturbances=clean.record(),checkpoint_audit=str(out/'checkpoint-audit.json'),
            execution_source='verified_frozen_training_snapshot',current_workspace_drift=proof['current_workspace_drift'])
        write(target/'plan.json',plan)
    elif args.mode=='report':inspect.report(target)
    else:inspect.render(target,out/'videos')
    write(out/(args.mode+'-frozen-execution.json'),dict(modules=LOADED,optimizer_updates=0))


if __name__=='__main__':main()
