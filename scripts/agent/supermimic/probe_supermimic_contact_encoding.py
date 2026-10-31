#!/usr/bin/env python3
"""Frozen SOLREF-input sensitivity on already sealed V9 real-action windows.

Replacing a real contact parameter by a training mean is deliberately a false
model input. It tests sensitivity only; it is not a corrected physical model,
a change to the simulator, a new policy evaluation, or evidence of task success.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import torch

from scripts.agent.supermimic import score_supermimic_physical_prediction as score
from ct_mjlab.supermimic.state import PHYS_DIM, SOLREF
from ct_mjlab.supermimic.networks import RunningNorm
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if ALGORITHM_CONTRACT != 'chuanliantui-supermimic-canonical-physics-path-derived-observation-reward-termination-v9':
        raise ValueError('Raw SOLREF mean replacement is V9-only; use the sealed V9 source/tools snapshot. V10 uses fixed physical K/B inputs.')
    torch.set_num_threads(2)
    args.baseline, args.output = args.baseline.resolve(), args.output.resolve()
    baseline_path = args.baseline/'predictions.pt'
    baseline_sha = score.digest(baseline_path)
    baseline_report = json.loads((args.baseline/'score.json').read_text())
    assert baseline_sha == baseline_report['predictions_sha256']
    artifact = torch.load(baseline_path, map_location='cpu', weights_only=False)
    current = score.current_training_sources()
    expected = score.contracts(score.command_cfg('walk'))
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for script in (Path(__file__).resolve(), Path(score.__file__).resolve()):
        relative = str(script.relative_to(score.ROOT))
        target = args.output/'snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, target)
        sources[relative] = score.digest(script)
    score.write(args.output/'script-sha256.json', sources)
    predictions, world_proofs = [], []
    parameter_copies = [w['truth']['parameters'].clone() for w in artifact['windows']]
    with ExitStack() as guards:
        guards.enter_context(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No probe optimizer')))
        guards.enter_context(patch.object(RunningNorm, 'update', side_effect=AssertionError('No probe normalization update')))
        for index, proof in enumerate(artifact['worlds']):
            label = ('midpoint', 'final')[index]
            model = score.load_world(label, Path(proof['checkpoint']), current, expected)
            assert model['proof']['checkpoint_sha256'] == proof['checkpoint_sha256']
            world = model['world']
            for recording, original in enumerate(artifact['windows']):
                for variant in ('solref_first_to_training_mean', 'solref_both_to_training_mean'):
                    parameters = original['truth']['parameters'].clone()
                    end = SOLREF.start+1 if variant == 'solref_first_to_training_mean' else SOLREF.stop
                    values = world.in_norm.mean[PHYS_DIM+SOLREF.start:PHYS_DIM+end]
                    parameters[..., SOLREF.start:end] = values
                    torch.testing.assert_close(parameters[..., :SOLREF.start],
                        original['truth']['parameters'][..., :SOLREF.start], atol=0, rtol=0)
                    torch.testing.assert_close(parameters[..., end:],
                        original['truth']['parameters'][..., end:], atol=0, rtol=0)
                    windows = dict(original, truth=dict(original['truth'], parameters=parameters))
                    predicted = score.predict(world, windows, 64)
                    predictions.append(dict(world_label=label, recording_index=recording,
                        variant=variant, replaced_parameter_indices=list(range(SOLREF.start, end)),
                        replacement_values=values.tolist(), tensors=predicted))
            model['proof']['world_state_sha256_after'] = score.state_digest(world)
            assert model['proof']['world_state_sha256_after'] == model['proof']['world_state_sha256_before']
            assert score.digest(Path(proof['checkpoint'])) == proof['checkpoint_sha256']
            world_proofs.append(model['proof'])
    sealed = args.output/'predictions.pt'
    torch.save(dict(baseline_path=str(baseline_path), baseline_sha256=baseline_sha,
        predictions=predictions, worlds=world_proofs, scope=__doc__), sealed)
    sealed_sha = score.digest(sealed)
    del predictions
    replay = torch.load(sealed, map_location='cpu', weights_only=False)
    results = []
    for item in replay['predictions']:
        # The labels, masses, windows and masks always remain the real ones.
        windows = artifact['windows'][item['recording_index']]
        summary = score.summarize(item['tensors'], windows)
        results.append(dict(world_label=item['world_label'], recording_index=item['recording_index'],
            variant=item['variant'], replaced_parameter_indices=item['replaced_parameter_indices'],
            replacement_values=item['replacement_values'], groups=summary['groups'],
            curve_24_policy_steps=summary['curve_24_policy_steps']))
    for original, before in zip(artifact['windows'], parameter_copies):
        assert torch.equal(original['truth']['parameters'], before)
    assert score.digest(baseline_path) == baseline_sha
    assert score.digest(sealed) == sealed_sha
    assert score.current_training_sources() == current
    for relative, sha in sources.items():
        assert score.digest(score.ROOT/relative) == sha
    result = dict(scope=__doc__, optimizer_steps=0, normalization_updates=0, simulator_steps=0,
        baseline=str(args.baseline), baseline_predictions_sha256=baseline_sha,
        original_recordings_unchanged=True, model_states_unchanged=True,
        probe_predictions_sha256=sealed_sha, worlds=world_proofs, results=results,
        source_sha256=sources,
        warning='False contact parameters are supplied only to the model; never treat these numbers as corrected nominal prediction or behavior acceptance.')
    score.write(args.output/'probe.json', result)
    print(json.dumps(dict(output=str(args.output), pairs=len(results),
        predictions_sha256=sealed_sha, optimizer_steps=0, simulator_steps=0)))


if __name__ == '__main__':
    main()
