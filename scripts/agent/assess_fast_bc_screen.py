#!/usr/bin/env python3
"""Check cached poses, frozen actions and motion statistics of the fast screen."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mjlab.utils.torch import configure_torch_backends
from ct_mjlab.bc import load_actor
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def assess(args):
    torch.set_num_threads(1);configure_torch_backends();actor,_=load_actor(args.checkpoint,'cuda:0')
    cached=torch.load(args.source/'nominal_mean/initial-state.pt',map_location='cpu',weights_only=False)
    groups={}
    for run in args.run:
        summary=json.loads((run/'comparison.json').read_text())
        assert summary['completed'] and summary['optimizer_updates']==0 and digest(args.checkpoint)==summary['checkpoint_sha256']
        assert summary['cached_initial_sha256']==digest(args.source/'nominal_mean/initial-state.pt')
        for name in summary['groups']:
            p=run/name;r=json.loads((p/'result.json').read_text());assert r['actor_unchanged'] and r['optimizer_updates']==0
            assert digest(p/'trace.npz')==r['trace_sha256']
            with np.load(p/'trace.npz') as archive:d={k:archive[k] for k in archive.files}
            indices=[row['original_env'] for row in r['rows']]
            np.testing.assert_array_equal(d['initial_qpos'][:,2:],cached['qpos'].numpy()[indices,2:])
            np.testing.assert_array_equal(d['initial_qvel'],cached['qvel'].numpy()[indices])
            np.testing.assert_array_equal(d['action'],d['mean']+d['noise'])
            for tick in np.linspace(0,len(d['time'])-1,8).round().astype(int):
                mean=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu().numpy()
                np.testing.assert_allclose(mean,d['mean'][tick],atol=2e-6,rtol=2e-6)
            noise_std=d['noise'].std((0,1));target=np.array([r['leg_std'],r['leg_std'],r['wheel_std'],r['leg_std'],r['leg_std'],r['wheel_std']])
            np.testing.assert_allclose(noise_std,target,atol=1e-6,rtol=.04)
            for row in r['rows']:
                i=row['env'];n=int(d['alive'][:,i].sum());late=d['time'][:n]>max(2.,r['seconds']-1 if r['seconds']<20 else 15.)
                if late.any():
                    leg=float(np.sqrt(np.square(d['motor_dq'][:n,i][:,[0,1,3,4]][late]).mean()))
                    speed=float(np.linalg.norm(d['features'][:n,i,4:6][late],axis=-1).mean())
                    assert abs(leg-row['leg_rms'])<1e-10 and abs(speed-row['speed'])<1e-10
            groups[name]=dict(strict_rise=r['strict_rise'],end_balance=r['end_balance'],qualified=r['qualified'],
                mean_leg_rms=r['mean_leg_rms'],max_leg_rms=max(x['leg_rms'] for x in r['rows'] if x['leg_rms'] is not None),
                mean_speed=r['mean_speed'],seconds=r['seconds'],num_envs=r['num_envs'],measured_action_noise_std=noise_std.tolist(),
                all_full_duration=bool(d['alive'].all()),leg_rms_within_002=sum(x['leg_rms'] is not None and x['leg_rms']<=.02 for x in r['rows']))
    result=dict(artifact_and_metric_checks_passed=True,optimizer_updates=0,groups=groups,
        checker_sha256=digest(__file__),checkpoint_sha256=digest(args.checkpoint),
        scope='Short screens are not20s acceptance; contact pass flags reused from actual-weight runtime probe. No retraining or reward attribution.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    for n in ('source','checkpoint','output'):p.add_argument('--'+n,type=Path,required=True)
    assess(p.parse_args())
