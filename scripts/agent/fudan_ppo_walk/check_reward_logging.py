"""Verify real TensorBoard events from the runner using fake collection and updates."""
import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ct_mjlab.fudan_ppo_walk import runner as module


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('ct_mjlab.fudan_ppo_walk.logging_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    def forbidden(*args, **kwargs):
        raise AssertionError('No real physical or optimizer step is permitted')

    captures = []
    with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
        for name in ('mj_step', 'mj_step1', 'mj_step2'):
            stack.enter_context(patch.object(mujoco, name, forbidden))
        stack.enter_context(patch.object(torch.optim.Adam, 'step', forbidden))
        for version, implementation in [('before', baseline), ('after', module)]:
            r = object.__new__(implementation.FudanRunner)
            r.log_dir = Path(temporary) / version
            r.writer = module.SummaryWriter(str(r.log_dir))
            r.iteration = r.transitions = r.primary_updates = r.encoder_updates = 0
            r.partial_update = False
            r.resume_from = None
            obs = torch.zeros(4, 25)
            critic = torch.zeros(4, 141)
            history = torch.zeros(4, 125)
            actions = torch.zeros(4, 6)
            done = torch.tensor([False, True, False, True])
            seen = []
            counter = 0
            terms = {name: torch.zeros(4) for name in module.WEIGHTS}
            positive = {'tracking_lin_vel', 'tracking_ang_vel', 'base_height'}
            coefficients = {name: (0. if name == 'collision' else (1. if name in positive else -1.) * (i + 1) * 1e-6)
                for i, name in enumerate(module.WEIGHTS)}

            def step(action):
                nonlocal counter
                counter += 1
                assert torch.equal(action, actions)
                # 每一步、每个环境都不同，且复用存储，以发现末步误记和引用累加错误。
                for name in terms:
                    terms[name].copy_(coefficients[name] * counter * torch.arange(1., 5.))
                reward = torch.stack(list(terms.values())).sum(0)
                return obs, critic, reward, done, {'terms': terms}, history

            def process(reward, done, info, obs):
                seen.append(reward.tolist())

            def update():
                r.primary_updates += 20
                r.encoder_updates += 20
                return [0., 0., 0., 0.]

            def save(name):
                path = r.log_dir / name
                path.write_text(str(r.iteration))
                return path

            r.save = save
            r.env = SimpleNamespace(num_envs=4, speed_limit=1., step=step)
            r.alg = SimpleNamespace(act=lambda *args: actions, process_env_step=process,
                compute_returns=lambda *args: None, update=update, learning_rate=.001)
            r.policy = SimpleNamespace(encode=lambda value: torch.zeros(4, 3))
            with patch.object(implementation, 'initialize_rollout', return_value=(obs, critic, history)), redirect_stdout(io.StringIO()):
                report = r.learn_until(float('inf'), max_iterations=2)
            assert counter == 96 and report['iterations'] == 2 and report['transitions'] == 384
            events = EventAccumulator(str(r.log_dir), size_guidance={'scalars': 0}).Reload()
            captures.append(seen)
            if version == 'before':
                assert not any(name.startswith('Rewards/') for name in events.Tags()['scalars'])
                continue
            tags = sorted(name for name in events.Tags()['scalars'] if name.startswith('Rewards/'))
            assert tags == sorted('Rewards/' + name for name in module.WEIGHTS)
            for name, coefficient in coefficients.items():
                values = events.Scalars('Rewards/' + name)
                assert [event.step for event in values] == [1, 2]
                for iteration, event in enumerate(values):
                    expected = coefficient * (24.5 + 48 * iteration) * 2.5
                    assert abs(event.value - expected) < 1e-8, (name, event.value, expected)
            totals = events.Scalars('Train/reward')
            for i, total in enumerate(totals):
                assert total.step == i + 1
                subtotal = sum(events.Scalars(tag)[i].value for tag in tags)
                assert abs(total.value - subtotal) < 1e-7
                last_step = sum(coefficients.values()) * (48 * (i + 1)) * 2.5
                assert abs(total.value - last_step) > 1e-5
        assert captures[0] == captures[1]
    package = Path(module.__file__).parent
    frozen = Path('logs/mjlab/fudan_ppo_walk/20261102_reset_v2_iter3000/source/ct_mjlab/fudan_ppo_walk')
    unchanged = ['env.py', 'contract.py', 'ppo.py', 'network.py', 'storage.py']
    for name in unchanged:
        assert hashlib.sha256((package / name).read_bytes()).digest() == hashlib.sha256((frozen / name).read_bytes()).digest()
    result = dict(passed=True, physics_steps=0, optimizer_updates=0, model_forwards=0,
        real_tensorboard_event_files_checked=True, reward_tags=tags, logged_iterations=[1, 2],
        aggregation='mean of weighted/clipped contributions over all48 steps and all environments',
        independent_expected_means=True, term_sum_matches_total=True, total_is_not_last_step=True,
        original_learning_inputs_unchanged=True, reward_and_algorithm_sources_unchanged=unchanged)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
