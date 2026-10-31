#!/usr/bin/env python3
"""Recompute heldout BC motion metrics and compare immutable initial states."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.bc import load_actor


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def assess(args):
    r=json.loads((args.run/'result.json').read_text());old=json.loads((args.previous/'result.json').read_text())
    assert r['completed'] and r['optimizer_updates']==0 and r['independent_student_control'] and not r['teacher_queries_only']
    ppo_path=getattr(args,'ppo_checkpoint',None)
    assert r['checkpoint_sha256']==digest(ppo_path or args.checkpoint) and r['trace_sha256']==digest(args.run/'trace.npz')
    assert old['trace_sha256']==digest(args.previous/'trace.npz')
    keys=lambda rows:[(x['group_id'],x['leg_delay_s'],x['wheel_delay_s']) for x in rows]
    assert keys(r['rows'])==keys(old['rows']) and len(r['rows'])==40
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    actor,saved=load_actor(args.checkpoint,'cuda:0')
    assert saved['total_actor_updates']==args.expected_updates==r['actor_updates']
    if ppo_path is not None:
        ppo=torch.load(ppo_path,map_location='cuda:0',weights_only=False)
        assert ppo['infos']['bc_init']['sha256']==digest(args.checkpoint)
        assert ppo['iter']+1==r['ppo_iterations']
        actor.load_state_dict(ppo['actor_state_dict'],strict=True)
    with np.load(args.run/'trace.npz') as archive,np.load(args.previous/'trace.npz') as previous:
        d={k:archive[k] for k in archive.files}
        np.testing.assert_array_equal(d['initial_qpos'],previous['initial_qpos'])
        np.testing.assert_array_equal(d['initial_qvel'],previous['initial_qvel'])
    wheel_error=None
    if saved['dataset'].get('observation_dim',39)>=40:
        import mujoco
        from ct_mjlab.envs.chuanliantui.model import MODEL_XML
        model=mujoco.MjModel.from_xml_path(str(MODEL_XML))
        adr=[model.joint(n).qposadr[0] for n in ('lfwheel','rfwheel')]
        # 用独立公式重算学生实际动作前的轮角差；仅检查首回合有效帧。
        pre=np.concatenate((d['initial_qpos'][None],d['qpos'][:-1]))
        delta=pre[:,:,adr]-d['initial_qpos'][None,:,adr]
        expected=.03*(delta[:,:,0]-delta[:,:,1]);valid=d['alive']
        np.testing.assert_allclose(d['observation'][:,:,39][valid],expected[valid],atol=3e-7,rtol=2e-6)
        wheel_error=float(np.max(np.abs(d['observation'][:,:,39][valid]-expected[valid])))
    reference_error=None
    angle_error=None
    if saved['dataset'].get('wheel_angles'):
        import mujoco
        from ct_mjlab.envs.chuanliantui.model import MODEL_XML
        model=mujoco.MjModel.from_xml_path(str(MODEL_XML))
        adr=[model.joint(n).qposadr[0] for n in ('lfwheel','rfwheel')]
        pre=np.concatenate((d['initial_qpos'][None],d['qpos'][:-1]))
        valid=d['alive']
        from ct_mjlab import wheel_angles
        if saved['dataset']['observation_contract']==wheel_angles.HANDOVER_OBSERVATION_CONTRACT:
            from ct_mjlab.bc import check_handover_wheel_pairs
            np.testing.assert_allclose(d['source_q'][...,[2,5]][valid],pre[:,:,adr][valid],atol=0,rtol=0)
            expected=check_handover_wheel_pairs(d,d['source_q'],d['observation'],valid).clip(-100,100)
        else:expected=(pre[:,:,adr]-d['initial_qpos'][None,:,adr]).clip(-100,100)
        np.testing.assert_allclose(d['observation'][:,:,14:16][valid],expected[valid],atol=2e-6,rtol=2e-6)
        angle_error=float(np.max(np.abs(d['observation'][:,:,14:16][valid]-expected[valid])))
        assert not d['actuator_torque'][:20].any()
    if saved['dataset'].get('observation_dim')==41:
        from ct_mjlab.bc import check_reference_pairs
        check_reference_pairs(d,d['observation'],d['alive'])
        reference_error=float(np.max(np.abs(d['observation'][:,:,40:41]-d['reference_error'])))
    assert not d['observation'][:,:,:3].any(),'Zero commands must remain zero'
    differences=[]
    for tick in np.linspace(0,len(d['time'])-1,16).round().astype(int):
        predicted=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu().numpy()
        np.testing.assert_allclose(predicted,d['action'][tick],atol=2e-6,rtol=2e-6)
        differences.append(float(np.max(np.abs(predicted-d['action'][tick]))))
    checked=[];kinematic=[]
    limits={'last5s_leg_velocity_rms_rad_s':.02,'last5s_mean_speed_m_s':.01,
        'last5s_xy_excursion_m':.02,'last5s_mean_abs_yaw_rate_rad_s':.02}
    for row in r['rows']:
        i=row['env'];length=int(d['alive'][:,i].sum());late=d['time'][:length]>15.
        f=d['features'][:length,i];q=d['qpos'][:length,i]
        metrics=dict(last5s_leg_velocity_rms_rad_s=float(np.sqrt(np.square(d['motor_dq'][:length,i][:,[0,1,3,4]][late]).mean())) if late.any() else None,
            last5s_mean_speed_m_s=float(np.linalg.norm(f[late,4:6],axis=-1).mean()) if late.any() else None,
            last5s_xy_excursion_m=float(np.linalg.norm(q[late,:2]-q[late,:2][0],axis=-1).max()) if late.any() else None,
            last5s_mean_abs_yaw_rate_rad_s=float(np.abs(f[late,9]).mean()) if late.any() else None)
        for k,v in metrics.items():
            assert (v is None and row[k] is None) or (v is not None and abs(v-row[k])<1e-10)
        expected=bool(row['strict_rise'] and row['final_balance_v31'] and row['full_20s'] and row['finite'] and
            all(v is not None and v<=limits[k] for k,v in metrics.items()))
        assert expected==row['qualified']
        if length>2:
            # 世界坐标差分独立核对水平速度，排除特征索引或坐标误读。
            actual=np.linalg.norm(np.diff(q[:,:2],axis=0)/.01,axis=-1)
            recorded=np.linalg.norm(d['qvel'][1:length,i,:2],axis=-1)
            stable=(d['time'][1:length]>2.) & (d['time'][1:length]<19.99)
            if stable.any():kinematic.append(float(np.max(np.abs(actual[stable]-recorded[stable]))))
        checked.append(row)
    stats={k:dict(limit=limit,passed=sum(x[k] is not None and x[k]<=limit for x in checked),
        min=min((x[k] for x in checked if x[k] is not None),default=None),
        max=max((x[k] for x in checked if x[k] is not None),default=None)) for k,limit in limits.items()}
    reference=json.loads(args.dataset.read_text())
    good={(e['group_id'],e['leg_delay_s'],e['wheel_delay_s']) for e in reference['episodes']}
    original_good=[x for x in checked if (x['group_id'],x['leg_delay_s'],x['wheel_delay_s']) in good]
    lost_balance=[new['env'] for before,new in zip(old['rows'],checked)
        if before['final_balance_v31'] and not new['final_balance_v31']]
    lost_rise=[new['env'] for before,new in zip(old['rows'],checked)
        if before['strict_rise'] and not new['strict_rise']]
    result=dict(passed=True,artifact_and_metric_checks_passed=True,optimizer_updates=0,identical_initial_states=True,
        teacher_not_used_for_actions=True,independent_actor_action_recomputed=True,max_actor_action_difference=max(differences),
        max_speed_difference_m_s=max(kinematic,default=None),
        old={k:old[k] for k in ('strict_rise','final_balance','qualified')},
        new={k:r[k] for k in ('strict_rise','final_balance','qualified')},full_20s=sum(x['full_20s'] for x in checked),
        no_rise_or_balance_regression=not lost_balance and not lost_rise,
        lost_previous_balance_cases=lost_balance,lost_previous_rise_cases=lost_rise,
        max_wheel_observation_error_m=wheel_error,
        max_wheel_angle_observation_error_rad=angle_error,
        max_reference_observation_error_m=reference_error,
        behavior_acceptance_passed=r['qualified']==40,late5s=stats,
        teacher_reference_cases=len(original_good),teacher_reference_successes_retained=sum(x['qualified'] for x in original_good),
        teacher_reference_successes_lost=[x['env'] for x in original_good if not x['qualified']],
        checkpoint_sha256=r['checkpoint_sha256'],trace_sha256=r['trace_sha256'],checker_sha256=digest(__file__),
        scope='Nominal independent40; recorded runtime v3.1 contact probe and recomputed motion/action metrics. No evaluation updates or hardware.',
        ppo_iterations=r.get('ppo_iterations'),smp_reward_weight=r.get('smp_reward_weight'))
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--expected-updates',type=int,default=15000)
    p.add_argument('--ppo-checkpoint',type=Path)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);assess(p.parse_args())
