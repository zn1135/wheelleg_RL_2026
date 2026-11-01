"""Frozen two-second encoder/controller crossover; diagnostic, not acceptance."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.supermimic import networks
from ct_mjlab.supermimic.learner import Learner
from ct_mjlab.supermimic.networks import RunningNorm

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to5200_r3'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parts-only', action='store_true', help='A5200 with A4200 legs or wheels on the same current observation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    # Preserve the original evaluator's capture, physical protocol and termination.
    sys.path.insert(0, str(ROOT/'scripts/agent'))
    spec = importlib.util.spec_from_file_location('_original_v13_assessment', ROOT/'scripts/agent/assess_chuanliantui_supermimic.py')
    assess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assess)
    checkpoints = {4200:RUN/'actor_004200.pt', 5200:RUN/'final.pt'}
    actors, hashes = {}, {}
    for n, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert Learner._saved_optimizer_counts(saved)['actor'] == n
        actors[n] = networks.make_actor('cuda:0').eval().requires_grad_(False)
        actors[n].load_state_dict(saved['actor'], strict=True)
        hashes[n] = assess.actor_digest(actors[n])
        assess.verify_training_sources(path, args.output, 'supermimic')
    results = []
    fixed_reference = None
    variants = [(4200,4200,None),(5200,5200,None),(5200,4200,None),(4200,5200,None)]
    if args.parts_only:
        variants = [(5200,5200,'legs'),(5200,5200,'wheels')]
    for controller, encoder, part in variants:
        label = f'controller{controller}-encoder{encoder}'+('' if part is None else '-old-'+part)
        histories, estimates, cross_actions = [], [], []
        def act_override(actor, history):
            # All four outputs here share exactly the same observed history.
            velocity = {n:a.encoder(history) for n,a in actors.items()}
            actions = {(c,e):actors[c].mlp(torch.cat((history[..., -25:],velocity[e]),-1))
                for c,e in ((4200,4200),(5200,5200),(5200,4200),(4200,5200))}
            histories.append(history.cpu().clone())
            estimates.append(torch.stack([velocity[n]/2 for n in (4200,5200)],1).cpu())
            cross_actions.append(torch.stack(list(actions.values()),1).cpu())
            result = actions[controller,encoder].clone()
            if part is not None:
                ids = [0,1,3,4] if part=='legs' else [2,5]
                # Hold the A5200 velocity estimate fixed to isolate the action channels.
                result[:,ids] = actions[4200,5200][:,ids]
            return result
        output = args.output/label
        config = SimpleNamespace(checkpoint=checkpoints[controller], policy_kind='supermimic',
            output=output, num_envs=96, seed=43, seconds=2., height=.26,
            speeds=[0.,.5,-.5], yaw_rates=None, conditions='nominal', device='cuda:0')
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training')), \
             patch.object(RunningNorm,'update',side_effect=AssertionError('No normalization update')), \
             patch.object(networks,'act',act_override):
            assess.run(config)
        assert all(assess.actor_digest(actors[n]) == hashes[n] for n in actors)
        summary_path = output/'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['diagnostic_override'] = dict(controller=controller, encoder=encoder,
            replacement_action_part=part, replacement_controller=4200 if part else None,
            original_act_bypassed=True, normal_acceptance_claim=False,
            encoder_velocity_trace_note='Original recorder reports controller checkpoint encoder; effective encoder stored in crossover.pt.')
        write(summary_path,summary)
        torch.save(dict(history=torch.stack(histories), encoder_velocity=torch.stack(estimates),
            actions=torch.stack(cross_actions), encoder_order=[4200,5200],
            action_order=[[4200,4200],[5200,5200],[5200,4200],[4200,5200]]),output/'crossover.pt')
        with np.load(output/'trajectory.npz') as z:
            fixed = {k:z[k].copy() for k in z.files if k.startswith('fixed_')}
        if fixed_reference is None:
            fixed_reference=fixed
        else:
            assert fixed.keys()==fixed_reference.keys()
            assert all(np.array_equal(v,fixed_reference[k]) for k,v in fixed.items())
        rows = summary['summary']['per_environment']
        results.append(dict(label=label, survived=sum(r['survived'] for r in rows),
            numerical_failures=summary['summary']['numerical_failures'],
            end_min_s=min(r['episode_end_s'] for r in rows), end_max_s=max(r['episode_end_s'] for r in rows),
            output=str(output), trajectory_sha256=summary['trajectory_sha256'],
            crossover_sha256=assess.digest(output/'crossover.pt')))
        print(json.dumps(results[-1]),flush=True)
    write(args.output/'result.json',dict(runs=results,initial_physical_arrays_equal=True,
        actor_parameters_unchanged=True,parameter_updates=0,normalization_updates=0,
        checkpoint_sha256={str(k):assess.digest(v) for k,v in checkpoints.items()},
        scope='2-second frozen diagnostic, seed43/96 envs. All requested velocities remain zero; not full behavior acceptance.',
        script_sha256=assess.digest(Path(__file__))))


if __name__=='__main__':
    main()
