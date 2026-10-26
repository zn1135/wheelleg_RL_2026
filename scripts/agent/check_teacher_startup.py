#!/usr/bin/env python3
"""Check 200ms teacher motor hold, fresh takeover and independent reset; no training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends

from ct_mjlab.teacher.env import make_teacher_env_cfg


@torch.inference_mode()
def check(args):
    configure_torch_backends()
    cfg = make_teacher_env_cfg(initial='ground', num_envs=20, seed=42,
                               velocity_integral=True, ground_start_hold_s=.2)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env.reset(seed=42)
        term = env.action_manager.get_term('teacher')
        command = torch.tensor([0., 0., .30], device=args.device).repeat(20, 1)
        assert not term.control_active.any()
        initial_height = env.sim.data.qpos[:, 2].clone()
        for _ in range(200):
            env.step(command)
            assert not term.torques.any() and not term.h7_torques.any()
            assert not env.sim.data.actuator_force[:, term.motor_actuators].any()
            assert not term.control_active.any()
            assert not term.controller.ready.any() and not term.velocity_ready.any()
            assert not term.controller.standup.elapsed.any()
        assert (env.sim.data.ctrl[:, term.spring_ids] > 0.).any()
        assert (env.sim.data.qpos[:, 2]-initial_height).abs().max() > .01
        env.step(command)
        assert term.control_active.all() and (term.torques.abs().amax(-1) > 1e-3).all()
        reset_ids = torch.tensor([0, 2], device=args.device)
        other = torch.tensor([i for i in range(20) if i not in (0, 2)], device=args.device)
        before = {key: value[other].clone() for key, value in vars(term.controller).items()
                  if isinstance(value, torch.Tensor)}
        ticks = term._control_ticks[other].clone()
        env.reset(env_ids=reset_ids)
        for key, value in before.items():
            torch.testing.assert_close(getattr(term.controller, key)[other], value, atol=0, rtol=0)
        torch.testing.assert_close(term._control_ticks[other], ticks, atol=0, rtol=0)
        assert not term.control_active[reset_ids].any() and not term._control_ticks[reset_ids].any()
        for _ in range(200):
            env.step(command)
            assert not term.control_active[reset_ids].any() and not term.torques[reset_ids].any()
            assert term.control_active[other].all()
        env.step(command)
        assert term.control_active.all() and (term.torques[reset_ids].abs().amax(-1) > 1e-3).all()
        report = dict(passed=True, num_envs=20, hold_s=.2, physics_dt=.001,
            zero_motor_requests_and_actuator_forces_first_200_ticks=True,
            spring_and_gravity_remain_active=True, controller_history_cleared_during_hold=True,
            first_active_tick=201, partial_reset_repeats_hold=True,
            other_environments_state_unchanged_on_reset=True, optimizer_updates=0,
            scope='real 402ms physics and partial reset; not complete standup/balance acceptance',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    check(parser.parse_args())


if __name__ == '__main__':
    main()
