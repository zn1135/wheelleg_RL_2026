#!/usr/bin/env python3
"""Audit qualified braking labels, measured stage entry and immutable heldout data."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,ORDER
from ct_mjlab.envs.chuanliantui.model import MODEL_XML,ACTUATED_JOINTS
from ct_mjlab.smp.features import inverse_rotate
from check_lqr_student_closed_loop import digest


def check(args):
    torch.set_num_threads(1)
    base=ActionDataset([args.dataset],args.corrections)
    mixed=ActionDataset([args.dataset],[*args.corrections,args.run])
    report=json.loads((args.run/'result.json').read_text());plan=json.loads((args.run/'plan.json').read_text())
    assert report['completed'] and report['optimizer_updates']==0 and not report['heldout_used']
    assert report['checkpoint_sha256']==plan['checkpoint_sha256']==digest(args.checkpoint)
    assert report['xml_sha256']==plan['xml_sha256']==digest(MODEL_XML)
    assert report['trace_sha256']==digest(args.run/'trace.npz')
    assert report['dataset_fingerprint']==base.description['fingerprint']
    for name,sha in report['source_sha256'].items():
        assert digest(args.run/'snapshot'/name)==sha
    train={e['group_id'] for e in base.motion.splits['train']}
    validation={e['group_id'] for e in base.motion.splits['validation']}
    assert set(plan['group_ids'])==train and not train&validation
    assert all(r['group_id'] in train for r in report['rows'])
    for x,y in zip(base.frames['validation'],mixed.frames['validation']):torch.testing.assert_close(x,y,atol=0,rtol=0)
    model=mujoco.MjModel.from_xml_path(str(MODEL_XML))
    qa=[model.joint(n).qposadr[0] for n in ACTUATED_JOINTS]
    va=[model.joint(n).dofadr[0] for n in ACTUATED_JOINTS]
    with np.load(args.run/'trace.npz') as archive:
        d={k:archive[k] for k in archive.files}
        o=d['observation'];applied=d['applied_action'];valid=d['label_valid'];label=d['label']
        selected=d['teacher_selected'];time=d['time'];entry=round(report['takeover_s']/.01)
        assert not valid[:entry].any() and not selected[:entry].any()
        accepted=np.array([r['qualified'] for r in report['rows']])
        assert not valid[:,~accepted].any()
        np.testing.assert_array_equal(valid,d['label_eligible'] & d['alive'] & accepted[None,:])
        assert np.all(d['teacher_phase'][valid]==3) and not d['teacher_fault'][valid].any()
        np.testing.assert_array_equal(applied[selected],label[selected])
        np.testing.assert_array_equal(applied[~selected],d['action'][~selected])
        np.testing.assert_allclose(o[1:,:,22:28],np.clip(applied[:-1,:,ORDER],-100,100),atol=2e-6,rtol=2e-6)
        assert not o[0,:,22:28].any()
        np.testing.assert_allclose(d['observation_time'],time-.01,atol=1e-8,rtol=0)
        np.testing.assert_array_equal(d['observation_time'],d['action_time'])
        preq=np.concatenate((d['initial_qpos'][None],d['qpos'][:-1]));prev=np.concatenate((d['initial_qvel'][None],d['qvel'][:-1]))
        np.testing.assert_allclose(d['source_q'],preq[:,:,qa],atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(d['source_dq'],prev[:,:,va],atol=2e-6,rtol=2e-6)
        position=(d['source_q']-d['nominal_motor_positions'])[:,:,ORDER];position[:,:,4:]=0.
        np.testing.assert_allclose(o[:,:,10:16],np.clip(position,-100,100),atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(o[:,:,16:22],np.clip(d['source_dq'][:,:,ORDER],-200,200)*.1,atol=2e-6,rtol=2e-6)
        g=torch.zeros(*preq.shape[:2],3);g[:,:,2]=-1
        gravity=inverse_rotate(torch.from_numpy(preq[:,:,3:7].copy()),g).numpy()
        np.testing.assert_allclose(o[:,:,7:10],gravity,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(o[:,:,4:7],np.clip(prev[:,:,3:6],-100,100)*.5,atol=2e-6,rtol=2e-6)
        assert np.isfinite(o[valid]).all() and np.isfinite(label[valid]).all()
        np.testing.assert_allclose(d['teacher_command'],np.broadcast_to([0.,0.,.30],d['teacher_command'].shape),atol=2e-8,rtol=0)
        assert not d['actual_torque'][:20].any()
        target=d['nominal_motor_positions'][[0,1,3,4]]+.5*label[valid][:,[0,1,3,4]]
        assert np.all(np.abs(target-d['reference_motor_positions'][[0,1,3,4]])<=3.14)
        assert np.all(np.abs(label[valid][:,[2,5]])<=15.)
        late=time>report['duration_s']-5.;post=time>report['takeover_s']
        for row in report['rows']:
            i=row['env'];entered=bool(selected[entry,i])
            assert row['checks']['entered_on_support']==entered
            if entered:
                assert row['entry_hold_substeps']==d['balance_hold'][entry-1,i]>=500
                assert row['old_teacher_phase']==d['teacher_phase'][entry-1,i]
                assert d['teacher_phase'][entry,i]==3
            f=d['features'][:,i];q=d['qpos'][:,i]
            rms=float(np.sqrt(np.square(d['motor_dq'][late,i][:,[0,1,3,4]]).mean()))
            speed=float(np.linalg.norm(f[late,4:6],axis=-1).mean());yaw=float(np.abs(f[late,9]).mean())
            drift=float(np.linalg.norm(q[late,:2]-q[late,:2][0],axis=-1).max())
            for k,v in [('last5s_leg_velocity_rms_rad_s',rms),('last5s_mean_speed_m_s',speed),
                        ('last5s_mean_abs_yaw_rate_rad_s',yaw),('last5s_xy_excursion_m',drift)]:
                assert abs(v-row[k])<1e-10
            if row['qualified']:
                assert row['checks'] and all(row['checks'].values())
                assert rms<=.02 and speed<=.01 and yaw<=.02 and drift<=.02
                assert d['balance_hold'][-1,i]>=500 and not d['done'][:,i].any()
                assert valid[:,i].sum()==post.sum()
        # 用动作生成前的实际状态统计新增覆盖，不能拿停车后的状态冒充制动输入。
        f=d['features'][:-1];mask=valid[1:]
        smooth=np.sqrt(np.square(f[:,:,[18,19,21,22]]).mean(-1))<=.05
        stable=(np.abs(f[:,:,0]-.33648)<=.03)&(f[:,:,3]<-.97)&smooth
        moving=np.linalg.norm(f[:,:,4:6],axis=-1)>.3;turning=np.abs(f[:,:,9])>.3
        coverage=dict(valid_teacher_frames=int(valid.sum()),stable_frames=int((mask&stable).sum()),
            moving_frames=int((mask&moving).sum()),turning_frames=int((mask&turning).sum()),
            moving_turning_frames=int((mask&moving&turning).sum()),
            stable_moving_frames=int((mask&stable&moving).sum()),
            stable_moving_turning_frames=int((mask&stable&moving&turning).sum()))
    added=len(mixed.frames['train'][0])-len(base.frames['train'][0])
    assert added==report['valid_correction_frames']==coverage['valid_teacher_frames']>0
    result=dict(passed=True,optimizer_updates=0,qualified_episodes=int(accepted.sum()),candidates=len(accepted),
        accepted_pose_groups=len({r['group_id'] for r in report['rows'] if r['qualified']}),
        added_training_frames=added,coverage=coverage,heldout_frames_unchanged=True,pre_step_pairs_checked=True,
        prefix_and_rejected_labels_excluded=True,stage_entry_on_measured_support=True,actual_teacher_actions_match_labels=True,
        source_snapshots_verified=True,checkpoint_sha256=report['checkpoint_sha256'],trace_sha256=report['trace_sha256'],
        augmented_fingerprint=mixed.description['fingerprint'],checker_sha256=digest(__file__),
        scope='Data integrity, fixed quality gates and loader compatibility; no new policy training or independent student acceptance.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',default=[]);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);check(p.parse_args())
