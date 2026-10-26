#!/usr/bin/env python3
"""Independently recompute frozen-BC action/noise and motion evidence; no learning."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.bc import load_actor
from mjlab.utils.torch import configure_torch_backends
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def assess(args):
    torch.set_num_threads(1);configure_torch_backends()
    comparison=json.loads((args.run/'comparison.json').read_text())
    assert comparison['completed'] and comparison['optimizer_updates']==0 and digest(args.checkpoint)==comparison['checkpoint_sha256']
    actor,_=load_actor(args.checkpoint,'cuda:0');std=actor.state_dict()['distribution.std_param'].cpu().numpy()
    summaries={};initials={};noise=[]
    for name,declared in comparison['groups'].items():
        directory=args.run/name;r=json.loads((directory/'result.json').read_text())
        assert r['optimizer_updates']==0 and r['actor_unchanged'] and r['independent_student_control'] and not r['teacher_queries_only']
        assert r['checkpoint_sha256']==comparison['checkpoint_sha256']
        assert digest(directory/'trace.npz')==r['trace_sha256']==declared['trace_sha256']
        assert digest(directory/'probe.npz')==r['probe_sha256']==declared['probe_sha256']
        initials[name]=torch.load(directory/'initial-state.pt',map_location='cpu',weights_only=False)
        with np.load(directory/'trace.npz') as d,np.load(directory/'probe.npz') as p:
            np.testing.assert_array_equal(d['action'],p['action_mean']+p['action_noise'])
            np.testing.assert_array_equal(d['initial_qpos'],initials[name]['qpos'].numpy())
            assert not d['actuator_torque'][:20].any() and len(r['rows'])==40
            for tick in np.linspace(0,len(d['time'])-1,16).round().astype(int):
                wanted=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu().numpy()
                np.testing.assert_allclose(wanted,p['action_mean'][tick],atol=2e-6,rtol=2e-6)
            actual_std=p['action_noise'].std((0,1))
            if r['action_sampling']:
                np.testing.assert_allclose(actual_std,std,atol=.001,rtol=0);noise.append(p['action_noise'].copy())
            else:assert not p['action_noise'].any()
            timing=[];sample_time=d['time'];support_hold=p['balance']
            for row in r['rows']:
                i=row['env'];n=int(d['alive'][:,i].sum());late=d['time'][:n]>15.
                assert row['strict_rise']==bool(p['strict'][:n,i].any())
                assert row['final_balance_v31']==bool(n==2000 and p['balance'][n-1,i]>=500)
                if late.any():
                    metrics={'last5s_leg_velocity_rms_rad_s':float(np.sqrt(np.square(d['motor_dq'][:n,i][:,[0,1,3,4]][late]).mean())),
                        'last5s_mean_speed_m_s':float(np.linalg.norm(d['features'][:n,i,4:6][late],axis=-1).mean()),
                        'last5s_xy_excursion_m':float(np.linalg.norm(d['qpos'][:n,i,:2][late]-d['qpos'][:n,i,:2][late][0],axis=-1).max())}
                    for k,v in metrics.items():assert abs(v-row[k])<1e-10,(name,i,k)
                bad=0;first=None
                for tick in range(n):
                    bad=bad+1 if sample_time[tick]>=2. and support_hold[tick,i]<5 else 0
                    if bad>=10:
                        first=float(sample_time[tick-9]);break
                timing.append(dict(env=i,first_strict_support_s=row['first_strict_support_s'],
                    first_sustained_support_violation_after2s_s=first,
                    terminated_before20s_s=n*.01 if n<2000 else None))
            def metric(k):
                v=[row[k] for row in r['rows'] if row[k] is not None]
                return dict(mean=statistics.mean(v) if v else None,count=len(v))
            periods={}
            for label,mask in [('before_push_2to4s',(d['time']>2.)&(d['time']<=4.)),('late5s',d['time']>15.)]:
                valid=d['alive'][mask];f=d['features'][mask][valid];q=d['motor_dq'][mask][valid]
                reward=p['reward'][mask][valid];names=p['reward_names'].tolist()
                periods[label]=dict(valid_steps=len(f),mean_speed_m_s=float(np.linalg.norm(f[:,4:6],axis=-1).mean()),
                    leg_velocity_rms_rad_s=float(np.sqrt(np.square(q[:,[0,1,3,4]]).mean())),
                    smp_score=float(reward[:,names.index('smp_motion')].mean()/.01),
                    anchor_error_mean_m=float(p['anchor_error'][mask][valid].mean()),
                    anchor_valid_fraction=float(p['anchor_valid'][mask][valid].mean()))
            summaries[name]=dict(strict_rise=r['strict_rise'],final_balance=r['final_balance'],qualified=r['qualified'],
                full20=sum(row['full_20s'] for row in r['rows']),action_noise_std=actual_std.tolist(),periods=periods,
                **{k:metric(k) for k in ('last5s_mean_speed_m_s','last5s_leg_velocity_rms_rad_s','late5s_mean_tilt_deg')},
                first_strict_support_count=sum(row['first_strict_support_s'] is not None for row in r['rows']),
                timing=timing,
                sustained_balance_loss_after_strict_count=sum(row['first_sustained_balance_loss_after_strict_s'] is not None for row in r['rows']))
    def equal(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        else:torch.testing.assert_close(a,b,atol=0,rtol=0)
    equal(initials['nominal_mean'],initials['nominal_sample']);equal(initials['random_mean'],initials['random_sample'])
    for value in initials.values():
        equal(value['qpos'],initials['nominal_mean']['qpos']);equal(value['qvel'],initials['nominal_mean']['qvel'])
    n=min(len(x) for x in noise);np.testing.assert_array_equal(noise[0][:n],noise[1][:n])
    result=dict(artifact_and_metric_checks_passed=True,optimizer_updates=0,behavior_acceptance_passed=False,
        frozen_checkpoint_sha256=comparison['checkpoint_sha256'],checker_sha256=digest(__file__),groups=summaries,
        scope='Single fixed40 cohort/seed. Initial matching and independent noise/action/motion checks; continuous physical contact flags reused from runtime probe, not recomputed from contact forces.')
    (args.run/'assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    assess(p.parse_args())
