"""Four independent, single-factor recovery experiments on the r14 baseline.

Run with python -m ct_mjlab.experiments.recovery_ablation train|check.
All groups retain the physical interface, assets, PPO and command sampling.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from ..reference_training import huananhu_mdp, huananhu_rewards
from ..reference_training.entry import manifest, runner_cfg, serializable
from ..reference_training.huananhu_env_cfg import make_huananhu_env_cfg
from ..reference_training.huananhu_runner import HuananhuOnPolicyRunner, INIT_CHECKPOINT_ENV
from ..reference_training.model import ACTUATED_JOINTS

GROUPS = {
    'baseline': {'factor': 'none'},
    'wide_height': {'factor': 'positive_height_kernel', 'denominator': .0049},
    'mode_reward': {'factor': 'recover_velocity_reward_schedule'},
    'recover_timeout': {'factor': 'recover_physical_failure_delay', 'steps': 250},
}
SCHEDULED_REWARDS = ('track_lin_vel_xy', 'track_lin_vel_xy_square',
                     'track_ang_vel_z', 'track_ang_vel_z_square', 'stand_still_lin_vel')


def failure_delay(counter, failed, recover):
    threshold = torch.where(recover, 250, 20)
    counter = torch.where(failed, torch.minimum(counter + 1, threshold), 0)
    return counter, counter >= threshold


def recover_failure(env):
    """Baseline failure and metrics, changing only the physical-failure delay."""
    state = env.command_manager.get_term('standup')
    robot = env.scene['robot']
    if not hasattr(state, 'motor_v_adr'):
        ids, _ = robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
        state.motor_v_adr = robot.indexing.joint_v_adr[ids]
    velocity = env.sim.data.qvel[:, state.motor_v_adr]
    linear, angular = robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b
    finite = torch.isfinite(env.sim.data.qpos).all(-1) & torch.isfinite(env.sim.data.qvel).all(-1)
    numerical = ~finite | (velocity.abs().amax(-1) > 500.) | (linear.abs().amax(-1) > 100.) | (angular.abs().amax(-1) > 200.)
    state.numerical_failure.copy_(numerical)
    gravity = robot.data.projected_gravity_b
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.asin(gravity[:, 0].clamp(-1., 1.))
    contact = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1) > 1.
    tilted = (roll.abs() > math.radians(40)) | (pitch.abs() > math.radians(40))
    state.failure_count, delayed = failure_delay(
        state.failure_count, (contact | tilted) & ~numerical, state.control_mode.recover)
    height = getattr(state, 'height_above_ground', robot.data.root_link_pos_w[:, 2])
    stable = ~contact & ~tilted & (height > .20)
    state.stable_time = torch.where(stable, state.stable_time + env.step_dt, 0.)
    state.metrics['ground_start'].copy_(state.ground_start)
    state.metrics['recovered'].copy_(torch.maximum(state.metrics['recovered'], (state.stable_time >= .5).float()))
    state.metrics['numerical_failure'].copy_(numerical)
    state.metrics['upright_fraction'] += stable.float() * env.step_dt / env.cfg.episode_length_s
    state.metrics['height_error'] += (height - state.command[:, 2]).abs() * env.step_dt / env.cfg.episode_length_s
    state.metrics['velocity_error'] += (linear[:, 0] - state.command[:, 0]).abs() * env.step_dt / env.cfg.episode_length_s
    if state.cfg.strict_recovery_metrics:
        wheel_force = env.scene['wheel_contact'].data.force_history.norm(dim=-1).amax(-1)
        strict = (~contact & (roll.abs() <= math.radians(15)) & (pitch.abs() <= math.radians(15))
                  & ((height - state.command[:, 2]).abs() <= .03)
                  & (wheel_force > 5.).all(-1) & ~numerical & ~delayed)
        state.strict_stable_time = torch.where(strict, state.strict_stable_time + env.step_dt, 0.)
        state.strict_recovered |= state.strict_stable_time >= .5
        state.metrics['strict_recovered'].copy_(state.strict_recovered)
        state.metrics['strict_stand_fraction'] += strict.float() * env.step_dt / env.cfg.episode_length_s
    if state.control_mode is not None:
        wheel_force = env.scene['wheel_contact'].data.force_history.norm(dim=-1).amax(-1)
        state.control_mode.advance(dict(height=height,roll=roll,pitch=pitch,
            base_contact=env.scene['base_contact'].data.force.norm(dim=-1).amax(-1),
            wheel_contact=wheel_force,valid=~numerical & ~delayed),
            state.command[:,2],env.common_step_counter)
    return delayed | numerical


def wide_height_reward(env, *, name, height_gate):
    state = env.command_manager.get_term('standup')
    height = getattr(state, 'height_above_ground', env.scene['robot'].data.root_link_pos_w[:, 2])
    value = torch.exp(-(height - state.command[:, 2]).clamp(-.15, .15).square() / .0049)
    if hasattr(state, 'jump_phase'):
        value = value.masked_fill(state.jump_phase > 0, 0.)
    return torch.nan_to_num(value, nan=0., posinf=0., neginf=0.).masked_fill(state.numerical_failure, 0.)


def mode_reward(env, *, name, height_gate):
    value = huananhu_rewards.reward(env, name=name, height_gate=height_gate)
    # The termination manager advances modes before reward computation. Use the
    # resulting mode consistently for all five terms and the next observation.
    recover = env.command_manager.get_term('standup').control_mode.recover
    return value.masked_fill(recover, 0.)


def make_env_cfg(group, num_envs=4096, seed=42, play=False):
    if group not in GROUPS:
        raise ValueError(f'Unknown group: {group}')
    cfg = make_huananhu_env_cfg(play=play)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if group == 'wide_height':
        cfg.rewards['track_height_exp_tight'].func = wide_height_reward
    elif group == 'mode_reward':
        for name in SCHEDULED_REWARDS:
            cfg.rewards[name].func = mode_reward
    elif group == 'recover_timeout':
        cfg.terminations['failure'].func = recover_failure
    return cfg


def experiment_manifest(group, cfg):
    return dict(interface=f'chuanliantui-recovery-ablation-v1-{group}',
                base=manifest(), group=group, settings=GROUPS[group],
                module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                environment=serializable(cfg))


class AblationVecEnv(RslRlVecEnvWrapper):
    def __init__(self, group, num_envs=4096, seed=42, device='cuda:0', play=False):
        torch.manual_seed(seed)
        cfg = make_env_cfg(group, num_envs, seed, play)
        super().__init__(ManagerBasedRlEnv(cfg, device=device), clip_actions=None)
        self.obs = self.get_observations()
        self.experiment = experiment_manifest(group, cfg)

    def step(self, actions):
        self.obs, reward, done, extra = super().step(actions)
        return self.obs, reward, done, extra


class AblationRunner(HuananhuOnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), 'interface': self.env.experiment,
                            'recovery_ablation': self.env.experiment})

    def load(self, path, *args, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('infos', {}).get('recovery_ablation') != self.env.experiment:
            raise ValueError('Ablation group, source or configuration mismatch')
        return super().load(path, *args, **kwargs)


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'check'))
    parser.add_argument('--group', required=True, choices=GROUPS)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--start-gate', type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1 or args.steps < 1:
        parser.error('Environment, iteration and step counts must be positive')
    if os.environ.get(INIT_CHECKPOINT_ENV):
        raise ValueError('Random-start ablation cannot initialize from an external checkpoint')
    env = AblationVecEnv(args.group, args.num_envs, args.seed, args.device)
    if args.command == 'check':
        numerical = 0
        for _ in range(args.steps):
            obs, reward, done, extra = env.step(torch.zeros(args.num_envs, 6, device=args.device))
            assert obs['actor'].shape == (args.num_envs, 35)
            assert obs['critic'].shape == (args.num_envs, 78)
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(reward).all()
            numerical += int(env.unwrapped.command_manager.get_term('standup').numerical_failure.sum())
        result = dict(group=args.group, steps=args.steps, num_envs=args.num_envs,
                      finite=True, numerical_failure_transitions=numerical)
        print('ABLATION_CHECK ' + json.dumps(result), flush=True)
        env.close()
        return
    if args.run_dir is None:
        parser.error('--run-dir is required for training')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    cfg = runner_cfg(args.seed)
    cfg.update(run_name=args.group, max_iterations=args.iterations)
    (args.run_dir / 'interface.json').write_text(json.dumps(env.experiment, indent=2) + '\n')
    (args.run_dir / 'environment.json').write_text(json.dumps(serializable(env.unwrapped.cfg), indent=2) + '\n')
    (args.run_dir / 'runner.json').write_text(json.dumps(cfg, indent=2) + '\n')
    # Environment construction consumes RNG. Reseed immediately before network
    # construction, so each group has the same actor, critic and optimizer start.
    torch.manual_seed(args.seed)
    runner = AblationRunner(env, cfg, str(args.run_dir), device=args.device)
    runner.save(str(args.run_dir / 'model_initial.pt'), infos={'checkpoint_kind': 'before_first_update'})
    initial = torch.load(args.run_dir / 'model_initial.pt', map_location='cpu', weights_only=False)
    state = env.unwrapped.command_manager.get_term('standup')
    ready = dict(group=args.group, pid=os.getpid(), seed=args.seed,
                 actor_sha256=tensor_digest(initial['actor_state_dict'].items()),
                 critic_sha256=tensor_digest(initial['critic_state_dict'].items()),
                 reset_sha256=tensor_digest([('qpos', env.unwrapped.sim.data.qpos),
                                            ('qvel', env.unwrapped.sim.data.qvel),
                                            ('command', state.command)]),
                 ground_start_count=int(state.ground_start.sum()),
                 optimizer_empty=not bool(initial['optimizer_state_dict']['state']),
                 learning_rate=runner.alg.learning_rate,
                 cuda_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                 cuda_reserved_mib=torch.cuda.memory_reserved() / 2**20)
    del initial
    (args.run_dir / 'ready.json').write_text(json.dumps(ready, indent=2) + '\n')
    print('ABLATION_READY ' + json.dumps(ready), flush=True)
    if args.start_gate is not None:
        deadline = time.monotonic() + 600
        while not args.start_gate.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Coordinator did not release training start within 600 seconds')
            time.sleep(.5)
    runner.learn(args.iterations, init_at_random_ep_len=False)
    env.close()
    print('ABLATION_COMPLETE ' + json.dumps({'group': args.group, 'iterations': args.iterations}), flush=True)


if __name__ == '__main__':
    main()
