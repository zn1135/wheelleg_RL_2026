"""Check immediate fallen-episode failure through the real task step; no physics."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import section_dict
from .reset_v2_fixture import fixture


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    def verify(name, value):
        checks[name] = bool(value)
        assert checks[name], name
    def forbidden(*a, **kw):
        raise AssertionError('No integration, network forward or update')
    with ExitStack() as stack:
        for name in ('mujoco.mj_step', 'mjlab.envs.ManagerBasedRlEnv.step', 'torch.nn.Linear.forward', 'torch.optim.Adam.step'):
            stack.enter_context(patch(name, side_effect=forbidden))
        env = fixture()
        env.cfg.env = SimpleNamespace(**section_dict(env.cfg.env))
        env.cfg.env.fail_ticks = 0
        env.robot.data.projected_gravity_b[0] = torch.tensor([1., 0., 0.])
        _, _, _, done, info, _ = env.step(torch.zeros(env.num_envs, 6), auto_reset=False, resample=False)
        verify('same_gravity_threshold', env.cfg.env.terminal_gravity_z == -.1)
        verify('fallen_only_immediate_failure', done.tolist() == [True, False, False, False])
        verify('failure_not_timeout_bootstrap', not info['time_outs'].any())
        env.step(torch.zeros(env.num_envs, 6), auto_reset=True, resample=False)
        verify('only_fallen_environment_reset', env.reset_calls[-1].tolist() == [0])
        env.env.sim.data.qvel[1, 0] = float('nan')
        try:
            env.step(torch.zeros(env.num_envs, 6), resample=False)
        except FloatingPointError:
            stopped = True
        else:
            stopped = False
        verify('nonfinite_guard_remains_error', stopped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=all(checks.values()), checks=checks,
        physics_steps=0, model_forwards=0, optimizer_updates=0), indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
