#!/usr/bin/env python3
"""Check the imported trained PPO through real train/play/check dispatch; never learns."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import ct_mjlab
from ct_mjlab.envs import task_registry
from ct_mjlab.scripts import train, play, check
from ct_mjlab.smp.ppo import SmpOnPolicyRunner
from check_ppo_finetune_resume import equal

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    assert Path(ct_mjlab.__file__).resolve().is_relative_to(ROOT)
    assert 'chuanliantui_stand' in task_registry.names()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert saved['bc_finetune']['completed_iterations'] == 400
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == '3d41097c93b5c22f26fefc17f03c330f54f236ebd2bc76537dcc0768e1f8307a'
    assert hashlib.sha256(args.prior.read_bytes()).hexdigest() == '1bff6d8bd8f9200090fd14b1883cbcb3b3c5fab809c020e0f80dcabb6b90eda4'
    reached = []
    original = task_registry.make_alg_runner

    def make_runner(*pos, **kwargs):
        kwargs['log_dir'] = args.output/'train-dispatch'
        return original(*pos, **kwargs)

    def before_sampling(runner, **kwargs):
        assert kwargs['num_learning_iterations'] == 1
        assert runner.alg.completed_iterations == 400
        assert runner.alg.optimizer.state
        assert runner.env.unwrapped.common_step_counter == saved['infos']['env_state']['common_step_counter']
        # 训练入口必须完整续载，而非只读出actor。
        restored = runner.alg.save()
        for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'bc_finetune'):
            equal(restored[key], saved[key])
        reached.append(True)

    base = dict(task='chuanliantui', seed=42, num_envs=2, device='cuda:0', smp_prior=args.prior)
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training allowed')):
        with patch.object(task_registry, 'make_alg_runner', side_effect=make_runner), \
                patch.object(SmpOnPolicyRunner, 'learn', before_sampling):
            train.train(SimpleNamespace(**base, iterations=1, run_name='import-check', resume=args.checkpoint,
                stand_curriculum_stage=None, bc_init=None, ppo_init=None, smp_weight=None,
                bc_finetune=False, phase_mix=False, wheel_std=None))
        assert reached == [True]
        check.check(SimpleNamespace(**{**base, 'num_envs': 20}, checkpoint=args.checkpoint, steps=32))

        class Viewer:
            def __init__(self, env, policy): self.env, self.policy = env, policy
            def run(self):
                with torch.inference_mode():
                    obs = self.env.get_observations()
                    for _ in range(4):
                        obs, reward, _, _ = self.env.step(self.policy(obs))
                        assert obs['actor'].shape == (2, 39)
                        assert obs['critic'].shape == (2, 82)
                        assert torch.isfinite(reward).all()

        with patch('mjlab.viewer.ViserPlayViewer', Viewer):
            play.play(SimpleNamespace(**base, checkpoint=args.checkpoint, viewer='viser',
                viser_port=None, knockdown_every=None))
    result = dict(passed=True, optimizer_updates=0, loaded_module=str(Path(ct_mjlab.__file__).resolve()),
        task_registry=list(task_registry.names()), real_train_full_resume=True,
        nonempty_adam_and_actor_critic_reference_exact=True, resumed_iterations=400,
        learn_intercepted_before_sampling=True, finite_envs=20, finite_steps=32,
        real_play_steps=4, viewer_boundary_mocked=True,
        scope='Real destination workspace imports, model/prior hashes, trained resume, finite physics and play policy. '
              'No learning; graphical viewer and full behavior assessed separately.')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
