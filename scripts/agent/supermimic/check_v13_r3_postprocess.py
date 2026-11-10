"""Finite CPU checks for r3 postprocessing guards and preserved failure pixels."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT/'artifacts/supermimic/supermimic-v13-continuation-20261101-r3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = ART/'postprocess.py'
    assert target.is_file(), 'The required postprocess.py has not been implemented'
    spec = importlib.util.spec_from_file_location('r3_postprocess', target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = dict(world=21056, encoder=5200, actor=5200, critic=5200)
    added = dict(world=12000, encoder=3000, actor=3000, critic=3000)
    result = dict(final_optimizer_counts=counts, additional_optimizer_counts=added,
        stop_reason='actor_update_limit', target_reached_exactly=True,
        termination_mode='actor_update_count_only', budget_kind='actual_actor_updates')
    audit = dict(all_checks_passed=True, checks=dict(completion_verified=True,
        final_summary=dict(actual_optimizer_counts=counts), additional_optimizer_updates=added))
    module.validate_training_evidence(result, audit, dict(returncode=0))
    rejected = []
    for field, value in [('actor', 5196), ('world', 21052)]:
        bad = copy.deepcopy(result)
        bad['final_optimizer_counts'][field] = value
        try:
            module.validate_training_evidence(bad, audit, dict(returncode=0))
        except AssertionError:
            rejected.append(field)
    assert rejected == ['actor', 'world'], 'Partial training must not be reported as completed'
    video = dict(decoded_frames=750, decoded_fps=25., decoded_duration_s=30.,
        first_episode_only=True, physics_steps_executed=0, policy_calls=0,
        parameter_updates=0, input_sha256=dict(checkpoint='a'*64))
    module.validate_video_record(video, 'a'*64)
    for field, value in [('physics_steps_executed', 1), ('decoded_frames', 749)]:
        bad = copy.deepcopy(video)
        bad[field] = value
        try:
            module.validate_video_record(bad, 'a'*64)
        except AssertionError:
            rejected.append(field)
    try:
        module.validate_video_record(video, 'b'*64)
    except AssertionError:
        rejected.append('wrong_checkpoint')
    assert len(rejected) == 5
    # The source failure bar and every original pixel must survive the label operation.
    source = np.full((80,128,3), 41, dtype=np.uint8)
    source[:20] = (107,37,37)
    labeled = module.label_frame(source, 'V13 actual actor=3200')
    assert labeled.shape == (112,128,3)
    assert np.array_equal(labeled[32:], source)
    report = dict(all_checks_passed=True, rejected_invalid_evidence=rejected,
        original_failure_pixels_preserved=True, model_forwards=0,
        physics_steps=0, optimizer_updates=0,
        scope='Synthetic CPU evidence/label checks; does not measure policy behavior or render physics')
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
