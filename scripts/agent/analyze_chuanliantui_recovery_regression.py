"""Paired ground-start counterfactuals: actor weights vs mode observation; no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import huananhu_v14_height as height_profile
from ct_mjlab.experiments import huananhu_v14_recovery as recovery_profile
from ct_mjlab.experiments import huananhu_v14_wheel_support as support_profile
from ct_mjlab.experiments.wheel_support_goals import STAGES, PARAMETERS
from ct_mjlab.policy import actor_model
from ct_mjlab.reference_training import huananhu_observations


def model(path, profile):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    profile.validate_checkpoint(saved)
    actor = actor_model().to('cuda:0').eval()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
    return actor, dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      completed_updates=saved['infos']['completed_updates'])


def actor_input(obs, mode):
    if mode == 'dynamic':
        return obs
    copied = obs.clone()
    copied['actor'][:, 28:35] = copied['actor'].new_tensor((1., 0., 0., 0., 0., 0., 0.))
    return copied


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--continued', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    actors = {}
    metadata = {}
    for label, path, profile in (('2000', args.source, height_profile), ('3000', args.continued, recovery_profile)):
        actors[label], metadata[label] = model(path, profile)
    torch.manual_seed(42)
    cfg = recovery_profile.make_env_cfg(64, 42, play=True)
    cfg.scene.sensors = support_profile.make_env_cfg(64, 42, play=True).scene.sensors
    cfg.commands['standup'].ground_start_probability = 1.
    cfg.commands['standup'].height_range = (.30, .30)
    cfg.commands['standup'].resampling_time_range = (1e6, 1e6)
    cfg.auto_reset = False
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device='cuda:0'), clip_actions=None)
    try:
        raw = env.unwrapped
        state = raw.command_manager.get_term('standup')
        state.curriculum.stage = 1
        initial = {k: v.clone() for k, v in dict(qpos=raw.sim.data.qpos, qvel=raw.sim.data.qvel,
                    command=state.command, origins=raw.scene.env_origins).items()}
        np.savez_compressed(args.output / 'initial-ground.npz', **{k: v.cpu().numpy() for k, v in initial.items()})
        initial_hash = hashlib.sha256(initial['qpos'].cpu().numpy().tobytes() + initial['qvel'].cpu().numpy().tobytes()).hexdigest()
        report = dict(models=metadata, learning=False, num_envs=64, seed=42, seconds=20.,
                      height_m=.30, criteria=PARAMETERS, initial_state_sha256=initial_hash,
                      intervention='only actor weights and actor mode slots28:35; common physical environment', cases={})
        ids = torch.arange(64, device=env.device)
        for weights, mode in (('2000', 'fixed'), ('2000', 'dynamic'), ('3000', 'fixed'), ('3000', 'dynamic')):
            raw.reset(env_ids=ids)
            raw.scene.reset(ids)
            raw.sim.data.qpos.copy_(initial['qpos'])
            raw.sim.data.qvel.copy_(initial['qvel'])
            state.sampled.copy_(initial['command'])
            state.command.copy_(initial['command'])
            state.ground_command.zero_()
            state.ground_start.fill_(True)
            state.control_mode.reset(ids, state.ground_start)
            state.goal_times.zero_()
            state.goal_achieved.zero_()
            state.goal_step = -1
            state.curriculum.stage = 1
            raw.sim.forward()
            raw.sim.sense()
            huananhu_observations.reset_observations(raw, ids)
            raw.observation_manager.reset(ids)
            raw.obs_buf = raw.observation_manager.compute(update_history=True)
            obs = env.get_observations()
            assert torch.equal(raw.sim.data.qpos, initial['qpos']) and torch.equal(raw.sim.data.qvel, initial['qvel'])
            assert torch.equal(raw.scene.env_origins, initial['origins'])
            with torch.no_grad():
                fixed_action = actors[weights](actor_input(obs, 'fixed'))
                dynamic_action = actors[weights](actor_input(obs, 'dynamic'))
                difference = (fixed_action - dynamic_action).abs()
            hold = torch.zeros(64, 3, device=env.device)
            success = torch.zeros_like(hold, dtype=torch.bool)
            tail = torch.zeros_like(hold)
            active = torch.ones(64, dtype=torch.bool, device=env.device)
            height = torch.zeros(64, device=env.device)
            wheels = torch.zeros(64, 2, device=env.device)
            other = torch.zeros_like(height)
            frames = [initial['qpos'].cpu().numpy().copy()]
            for step in range(1000):
                with torch.no_grad():
                    action = actors[weights](actor_input(obs, mode))
                    obs, _, _, _ = env.step(torch.where(active[:, None], action, 0.))
                active &= ~state.numerical_failure
                values = support_profile.goals(raw)
                flags = torch.stack([values[s] for s in STAGES], -1) & active[:, None]
                hold = torch.where(flags, hold + raw.step_dt, 0.)
                success |= hold >= PARAMETERS['success_hold_s']
                if step >= 750:
                    tail += flags
                    height += raw.scene['robot'].data.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                    wheels += support_profile.wheel_support_force(raw)
                    other += support_profile.nonwheel_support_force(raw)
                if (step + 1) % 5 == 0:
                    frames.append(raw.sim.data.qpos.cpu().numpy().copy())
            label = weights + '_' + mode
            result = dict(initial_states_identical=True, numerical_failures=int((~active).sum()),
                successes={s: int((success[:, i] & active).sum()) for i, s in enumerate(STAGES)},
                sustained_last5s={s: int(((tail[:, i] >= 237.5) & active).sum()) for i, s in enumerate(STAGES)},
                balanced_samples=torch.nonzero((tail[:, 1] >= 237.5) & active).flatten().cpu().tolist(),
                initial_action_mean_abs_difference_per_channel=difference.mean(0).cpu().tolist(),
                initial_sample0_fixed_action=fixed_action[0].cpu().tolist(),
                initial_sample0_dynamic_action=dynamic_action[0].cpu().tolist(),
                sample0=dict(height_m=float(height[0]/250), wheel_support_n=(wheels[0]/250).cpu().tolist(),
                             nonwheel_support_n=float(other[0]/250)))
            report['cases'][label] = result
            np.savez_compressed(args.output / f'{label}-ground-qpos.npz', qpos=np.array(frames),
                                time_s=np.arange(len(frames)) * .1)
            (args.output / 'paired-regression.json').write_text(json.dumps(report, indent=2) + '\n')
            print('PAIRED_CASE ' + json.dumps({label: result}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
