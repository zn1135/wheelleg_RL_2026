#!/usr/bin/env python3
"""Generate train-only successful-actor rise and local-feedback retention pairs."""
import argparse,copy,json,math,shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,load_actor
from ct_mjlab.bc_preservation import migrate_wheel_actor,SCHEMA
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT
from check_lqr_student_closed_loop import digest
import eval_bc


@torch.inference_mode()
def prepare(args):
    torch.set_num_threads(1)
    data=ActionDataset([args.dataset],require_reference=True)
    actor,saved=load_actor(args.checkpoint,'cuda:0')
    actor=migrate_wheel_actor(actor,saved,data.description)
    before={k:v.clone() for k,v in actor.state_dict().items()}
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=copy.deepcopy(saved);runtime['dataset']=data.description
    call=SimpleNamespace(dataset=args.dataset,corrections=[],checkpoint=args.checkpoint,pose_dataset=args.pose_dataset,
        device='cuda:0',split='train',collect_corrections=False,full_precision=False,output=args.output/'rise-rollout')
    rise_run=getattr(args,'rise_run',None) or args.output/'rise-rollout'
    if getattr(args,'rise_run',None) is None:
        with patch.object(eval_bc,'load_actor',return_value=(actor,runtime)):eval_bc.run(call)
    report_path=rise_run/'result.json';report=json.loads(report_path.read_text())
    assert report['checkpoint_sha256']==digest(args.checkpoint) and report['split']=='train'
    assert report['base_motion_fingerprint']==data.motion.fingerprint and report['trace_sha256']==digest(rise_run/'trace.npz')
    if getattr(args,'rise_run',None) is None:
        report.update(diagnostic_only=True,original_training_dataset_fingerprint=saved['dataset']['fingerprint'],
            actor_transform='Legacy39 first-layer columns14,15 zero; all other weights retained',
            dataset_fingerprint_scope='Runtime train pose/observation manifest only')
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    train=set(data.description['group_split']['train']);heldout=set(data.description['group_split']['validation'])
    assert {r['group_id'] for r in report['rows']}==train and not train&heldout
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    with np.load(rise_run/'trace.npz') as d:
        success=np.array([r['strict_rise'] and r['final_balance_v31'] and r['full_20s'] for r in report['rows']])
        before_handover=(~d['wheel_angle_handover_latched'] if data.description['observation_contract']==HANDOVER_OBSERVATION_CONTRACT
            else ((d['time']-.01)<2.)[:,None])
        valid=d['alive']&success[None,:]&before_handover
        rise_observation=d['observation'][valid].copy();rise_target=d['action'][valid].copy()
        group=np.asarray([r['group_id'] for r in report['rows']]);rise_group=np.broadcast_to(group,valid.shape)[valid].copy()
        assert not d['actuator_torque'][:20].any()
        live_differences=[]
        for tick in range(200):
            actual=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu().numpy()
            np.testing.assert_allclose(actual,d['action'][tick],atol=2e-6,rtol=2e-6)
            live_differences.append(float(abs(actual-d['action'][tick]).max()))
    # 真实控制按完整环境批次复算；训练保留目标另外统一为IEEE FP32。
    configure_torch_backends(allow_tf32=False)
    plus=[];minus=[];groups=[];feedback_sources=[]
    for episode in data.motion.splits['train']:
        assert episode['group_id'] in train
        with np.load(episode['path']) as d:
            obs=d['observation'][[500,750,1000,1250,1500,1750]].copy()
        for variable in ('pitch','wheel_speed'):
            a=obs.copy();b=obs.copy()
            if variable=='pitch':
                roll=np.arctan2(-obs[:,8],-obs[:,9]);pitch=np.arcsin(obs[:,7].clip(-1,1))
                for value,sign in ((a,1),(b,-1)):
                    angle=pitch+sign*math.radians(.5)
                    value[:,7]=np.sin(angle);value[:,8]=-np.sin(roll)*np.cos(angle);value[:,9]=-np.cos(roll)*np.cos(angle)
            else:
                a[:,20]+=.025;a[:,21]-=.025;b[:,20]-=.025;b[:,21]+=.025
            plus.append(a);minus.append(b);groups.extend([episode['group_id']]*len(obs))
        feedback_sources.append(dict(episode_id=episode['episode_id'],sha256=episode['sha256'],group_id=episode['group_id']))
    feedback_plus=np.concatenate(plus);feedback_minus=np.concatenate(minus)
    def predict(obs):
        return torch.cat([actor.mlp(chunk.cuda()).cpu() for chunk in torch.from_numpy(obs).split(4096)])
    rise_actual_action=rise_target
    rise_target=predict(rise_observation).numpy()
    canonical_difference=float(abs(rise_target-rise_actual_action).max())
    delta=predict(feedback_plus)-predict(feedback_minus)
    rise_scale=torch.from_numpy(rise_target).std(0,correction=0).clamp_min(.1)
    feedback_scale=delta.square().mean(0).sqrt().clamp_min(.01)
    pairs=args.output/'pairs.npz'
    np.savez_compressed(pairs,rise_observation=rise_observation,rise_target=rise_target,rise_actual_action=rise_actual_action,rise_group=rise_group,
        feedback_plus=feedback_plus,feedback_minus=feedback_minus,feedback_target_delta=delta.numpy(),
        feedback_group=np.asarray(groups),rise_scale=rise_scale.numpy(),feedback_scale=feedback_scale.numpy())
    root=Path(__file__).resolve().parents[2];sources={}
    for file in list((root/'mjlab_training/src/ct_mjlab').rglob('*.py'))+[
        Path(__file__),root/'scripts/agent/eval_bc.py',root/'scripts/agent/check_lqr_student_closed_loop.py',
        root/'scripts/agent/check_smp_policy_source.py',root/'scripts/agent/collect_lqr_student.py']:
        name=file.relative_to(root);target=args.output/'snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target);sources[str(name)]=digest(file)
    manifest=dict(schema=SCHEMA,path='pairs.npz',sha256=digest(pairs),optimizer_updates=0,
        checkpoint_sha256=digest(args.checkpoint),initial_actor_updates=saved['total_actor_updates'],
        base_motion_fingerprint=data.motion.fingerprint,observation_contract=data.description['observation_contract'],
        xml_sha256=data.motion.xml_sha256,group_split=data.description['group_split'],
        rise_frames=len(rise_target),rise_successful_episodes=int(success.sum()),rise_pose_groups=len(set(rise_group)),
        feedback_pairs=len(delta),feedback_pose_groups=len(set(groups)),heldout_used=False,
        rise_source_trace_sha256=report['trace_sha256'],rise_source_report_sha256=digest(report_path),
        rise_source_run=str(rise_run),canonical_precision='ieee_fp32',
        max_actual_full_batch_action_difference=max(live_differences),
        max_canonical_fp32_vs_executed_tf32_difference=canonical_difference,
        feedback_sources=feedback_sources,source_sha256=sources,
        constraints=dict(rise='frozen stable actor IEEE FP32 outputs on actual successful rollout states before2s; executed actions stored separately',
            feedback='old actor plus-minus response only; pitch +/-0.5deg, wheel forward +/-0.25rad/s; absolute parking action free'),
        scope='Training poses only. Feedback pairs are synthetic input constraints, not physical perturbation demonstrations or SMP positives.')
    if data.description['observation_contract']==HANDOVER_OBSERVATION_CONTRACT:
        manifest['label_reference']=data.description['label_reference']
        manifest['constraints']['rise']='frozen actor outputs before shared recovery handover only; do not retain old post-rise drift actions'
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for k,v in before.items():torch.testing.assert_close(v,actor.state_dict()[k],atol=0,rtol=0)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_sha256','feedback_sources','group_split')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','pose-dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rise-run',type=Path,help='Reuse a verified completed train-only rollout; never reuses heldout states')
    prepare(p.parse_args())
