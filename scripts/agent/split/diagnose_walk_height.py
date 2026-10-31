#!/usr/bin/env python3
"""Compare zero/backward requests from matched states using a frozen walk policy."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts.agent.split.diagnose_walk_v8_regression import digest, run, write


def summarize(folder, labels):
    reports, arrays, traces = [], [], []
    for label in labels:
        meta = json.loads((folder/(label+'.json')).read_text())
        with np.load(folder/(label+'.npz')) as saved:
            fixed = {key:saved['fixed_'+key] for key in
                     ('qpos','qvel','nominal','kp','kd','strength','mass','friction')}
            values = saved['values']
        if arrays:
            assert all(np.array_equal(fixed[key], arrays[0][key]) for key in fixed)
        arrays.append(fixed)
        assert meta['actor_unchanged'] and meta['parameter_updates'] == 0
        def get(key):
            lo,hi = meta['layout'][key]
            return values[...,lo:hi]
        times = (np.arange(len(values))+1)*meta['dt']
        valid = get('valid')[...,0] > .5
        last = valid & (times[:,None] >= meta['seconds']-2.)
        height = get('height')[...,0]
        target = get('applied_command')[...,2]
        assert np.allclose(target[valid], meta['height'], rtol=0., atol=1e-7)
        per_env=[]
        for env_id,end in enumerate(meta['episode_end']):
            live = last[:,env_id]
            per_env.append(dict(env=env_id, survived=end>=meta['seconds']-1e-6,
                height_mean_m=None if not live.any() else float(height[live,env_id].mean()),
                height_mae_m=None if not live.any() else float(abs(height-target)[live,env_id].mean())))
        def mean(key):
            return get(key)[last].mean(0).tolist() if last.any() else None
        reports.append(dict(label=label, checkpoint=meta['checkpoint'],
            checkpoint_sha256=meta['checkpoint_sha256'],
            height_target_m=meta['height'],
            survived=sum(e['survived'] for e in per_env), per_environment=per_env,
            height_mean_m=None if not last.any() else float(height[last].mean()),
            height_mae_m=None if not last.any() else float(abs(height-target)[last].mean()),
            leg_length_mean_m=mean('leg_length'), swing_mean_deg=mean('swing'),
            leg_target_mean_rad=mean('leg_target'), joint_q_mean_rad=mean('joint_q'),
            torque_mean_Nm=mean('actual_torque'), support_mean_N=mean('support'),
            true_velocity_mean_m_s=mean('true_velocity'),
            tilt_mean_deg=mean('tilt'),
            summary=meta['summary']['cases']))
        traces.append(dict(height=height, valid=valid, times=times))
    assert reports[0]['checkpoint_sha256'] == reports[1]['checkpoint_sha256']
    delta=[]
    for zero,backward in zip(reports[0]['per_environment'],reports[1]['per_environment'],strict=True):
        assert zero['env'] == backward['env']
        measured=zero['height_mean_m'] is not None and backward['height_mean_m'] is not None
        delta.append(dict(env=zero['env'], both_survived=zero['survived'] and backward['survived'],
            backward_minus_zero_height_m=backward['height_mean_m']-zero['height_mean_m'] if measured else None,
            zero_height_mae_le3cm=measured and zero['height_mae_m'] <= .03,
            backward_height_mae_gt3cm=measured and backward['height_mae_m'] > .03))
    before=traces[0]['times'] <= 2.
    return dict(verified=True,parameter_updates=0,initial_and_physical_arrays_equal=True,
        protocol=f"same checkpoint/seed43/16 states/{reports[0]['height_target_m']}m/20s/2s warmup/2m/s2; only requested vx differs",
        pre_request_max_paired_height_difference_m=float(abs(traces[0]['height'][before]-traces[1]['height'][before]).max()),
        paired_height_deltas=delta, runs=reports,
        hypothesis_supported_pairs=sum(e['both_survived'] and e['zero_height_mae_le3cm']
            and e['backward_height_mae_gt3cm'] for e in delta),
        scope='Nominal command intervention only; supports behavioral association with requested backward motion, not a unique reward/learning cause or full-task acceptance')


def main(args):
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[3]
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    training=args.training_artifact.resolve()
    launch=json.loads((training/'launch.json').read_text())
    hashes=json.loads((training/'source-sha256.json').read_text())
    assets=json.loads((training/'physical-files-sha256.json').read_text())
    checkpoint=args.checkpoint.resolve()
    write(args.output/'experiment-card.json',dict(
        hypothesis='Compared with zero request, backward motion lowers steady body height at the same height target',
        single_variable='requested vx: 0 versus -1m/s', checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), seed=43, num_envs=16, seconds=20, height_m=args.height,
        acceleration_m_s2=2., warmup_s=2., training_updates=0,
        criterion='Both survive 20s; paired final2s height MAE <=3cm at zero and >3cm backward supports the hypothesis; report all 16 pairs and motor/geometry/load, do not claim unique cause',
        unchanged=['policy','initial and physical state','height','PD','reward','termination','noise/delay settings']))
    source=args.output/'frozen-runtime/mjlab_training/src/ct_mjlab'
    shutil.copytree(Path(launch['log_dir'])/'snapshot/ct_mjlab',source,
                    ignore=shutil.ignore_patterns('__pycache__'))
    (args.output/'frozen-runtime/resources').symlink_to(root/'resources',target_is_directory=True)
    assert all(digest(source/name)==sha for name,sha in hashes.items())
    assert all(digest(root/name)==sha for name,sha in assets.items())
    sys.path.insert(0,str(source.parent))
    from ct_mjlab.envs import task_registry
    args.seed,args.seconds,args.warmup = 43,20.,2.
    args.velocity_source,args.action_source = 'encoder','original'
    args.previous=checkpoint
    reference=None
    for label,speed in (('zero',0.),('backward',-1.)):
        cases=[dict(name=label,speed=speed,acceleration=2.,env_ids=list(range(16)))]
        _,fixed=run(args,checkpoint,label,cases,reference,task_registry)
        reference=fixed
    assert all(digest(root/name)==sha for name,sha in assets.items())
    report=summarize(args.output,('zero','backward'))
    report.update(source_files=len(hashes),resource_files=len(assets))
    write(args.output/'result.json',report)
    shutil.copy2(__file__,args.output/'source.py')
    shutil.copy2(root/'scripts/agent/split/diagnose_walk_v8_regression.py',args.output/'recorder-source.py')
    print(json.dumps(dict(verified=True,hypothesis_supported_pairs=report['hypothesis_supported_pairs'],
        runs=[dict(label=r['label'],survived=r['survived'],height_mean_m=r['height_mean_m'])
              for r in report['runs']]),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--height',type=float,default=.26)
    parser.add_argument('--device',default='cuda:0')
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Diagnosis cannot train')):
        main(parser.parse_args())
