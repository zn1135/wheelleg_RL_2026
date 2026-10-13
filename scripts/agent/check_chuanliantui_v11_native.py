"""Compare native V11 observations and controls with actual mjlab managers.

No learning or behavioral acceptance. Uses one GPU environment and tests
asymmetric state/action channels, clipping, springs, and ground hold boundaries.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'sim2sim'), str(ROOT / 'mjlab_training/src')]
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from ct_mjlab.reference_training.huananhu_env_cfg import make_huananhu_env_cfg
from ct_mjlab.reference_training import huananhu_observations
from mj_sim2sim_v11 import Playback, MOTORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = make_huananhu_env_cfg(play=True, dynamic_modes=False, policy_joint_order=MOTORS)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    cpu = Playback()
    rng = np.random.default_rng(42)
    maxima = dict(observation=0., motor_torque=0., spring_force=0.)
    try:
        env.reset(seed=42)
        action = env.action_manager.get_term('pd')
        state = env.command_manager.get_term('standup')
        obs = huananhu_observations.get_state(env)
        for probe in range(32):
            cpu.reset('ground' if probe >= 14 and probe % 2 else 'normal')
            cpu.steps = [0, 9, 10, 20][probe % 4]
            cpu.command[:] = [.4, -.3, .30]
            cpu.last_action[:] = rng.uniform(-150, 150, 6)
            cpu.data.qpos[cpu.q] += rng.uniform(-.3, .3, 6)
            cpu.data.qvel[:] = rng.uniform(-300, 300, cpu.model.nv)
            quat = rng.normal(size=4)
            cpu.data.qpos[3:7] = quat / np.linalg.norm(quat)
            if probe < 14:
                # Unsaturated unit probes must also detect swapped motor channels.
                cpu.data.qpos[cpu.q] = cpu.nominal
                cpu.data.qvel[:] = 0.
                cpu.last_action[:] = 0.
                if probe < 12:
                    cpu.last_action[probe % 6] = 1. if probe < 6 else -1.
            # Both models contain exactly the same named articulated joints.
            for j in range(cpu.model.njnt):
                cj = cpu.model.joint(j)
                gj = env.sim.mj_model.joint('robot/' + cj.name)
                nq, nv = (7, 6) if int(cpu.model.jnt_type[j]) == 0 else (1, 1)
                cqa, cva = int(cj.qposadr[0]), int(cj.dofadr[0])
                gqa, gva = int(gj.qposadr[0]), int(gj.dofadr[0])
                env.sim.data.qpos[0, gqa:gqa+nq] = torch.tensor(cpu.data.qpos[cqa:cqa+nq], device=env.device)
                env.sim.data.qvel[0, gva:gva+nv] = torch.tensor(cpu.data.qvel[cva:cva+nv], device=env.device)
            state.command[:] = torch.tensor(cpu.command, device=env.device)
            state.ground_start[:] = cpu.ground
            env.episode_length_buf[:] = cpu.steps
            values = torch.tensor(cpu.last_action[None], device=env.device)
            action.process_actions(values)
            obs.reset(torch.tensor([0], device=env.device))
            expected = obs.actor()[0].cpu().numpy()
            got = cpu.observation()
            maxima['observation'] = max(maxima['observation'], float(np.max(np.abs(got - expected))))
            np.testing.assert_allclose(got, expected, atol=2e-6, rtol=1e-6)
            action.apply_actions()
            torque, spring = cpu.controls(cpu.last_action)
            expected_torque = action.torques[0].cpu().numpy()
            expected_spring = env.sim.data.ctrl[0, action._spring_ids].cpu().numpy()
            maxima['motor_torque'] = max(maxima['motor_torque'], float(np.max(np.abs(torque - expected_torque))))
            maxima['spring_force'] = max(maxima['spring_force'], float(np.max(np.abs(spring - expected_spring))))
            np.testing.assert_allclose(torque, expected_torque, atol=1e-4, rtol=1e-6)
            np.testing.assert_allclose(spring, expected_spring, atol=1e-4, rtol=1e-6)
        record = dict(passed=True, probes=32, max_abs_error=maxima,
                      scope='interface only; not behavioral acceptance')
        with args.output.open('x') as f:
            json.dump(record, f, indent=2)
        print(json.dumps(record), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
