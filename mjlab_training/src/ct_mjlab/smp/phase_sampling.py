"""Physically stagger normal ground episodes before PPO collection starts."""
from pathlib import Path
import torch

CONTRACT = 'chuanliantui-physical-phase-mix-v1'


@torch.inference_mode()
def prepare(runner):
    raw = runner.env.unwrapped
    steps = raw.max_episode_length
    if steps != 2000 or raw.episode_length_buf.any():
        raise ValueError('Phase mixing requires fresh normal20s episodes')
    learning_counter = raw.common_step_counter
    # 每个环境真正复位后再走到目标时刻，不伪造回合年龄或跳过电机等待。
    age = torch.div(torch.arange(raw.num_envs, device=raw.device)*steps,
                    raw.num_envs, rounding_mode='floor')
    age = age[torch.randperm(raw.num_envs, device=raw.device)]
    scheduled = {}
    for i, when in enumerate((steps-age).cpu().tolist()):
        scheduled.setdefault(when, []).append(i)
    obs = runner.env.get_observations()
    physical_failures = 0
    for tick in range(1, steps+1):
        before = raw.episode_length_buf.clone()
        action = runner.alg.actor(obs, stochastic_output=True)
        obs, _, _, _ = runner.env.step(action)
        physical_failures += int(raw.termination_manager.terminated.sum())
        # 相同的0.2s门控在每次真实复位之后都必须生效。
        assert not raw.action_manager.get_term('pd').torques[before < 20].any()
        if tick in scheduled:
            ids = torch.tensor(scheduled[tick], device=raw.device, dtype=torch.long)
            raw.reset(env_ids=ids)
            obs = runner.env.get_observations()
        if tick % 500 == 0:
            print(f'Phase mixing: simulated {tick}/{steps} steps; no optimizer updates', flush=True)
    actual_age = raw.episode_length_buf.clone()
    # common_step_counter继续表示已训练的策略步；额外物理步另存账本。
    # 各环境自己的episode_length_buf以及观测/PD/SMP历史全部保留。
    raw.common_step_counter = learning_counter
    if hasattr(raw, '_isaac_standup_rewards'):
        raw._isaac_standup_rewards.step = -1
    stats = runner.phase_mixing
    stats['prelude_runs'] += 1
    stats['prelude_control_steps_total'] += steps
    stats['last_prelude'] = dict(steps=steps, physical_failures=physical_failures,
        completed_training_iterations=runner.alg.completed_iterations,
        target_age_steps=age.cpu().tolist(), actual_age_steps=actual_age.cpu().tolist(),
        phase_counts=[int((actual_age < 20).sum()),
                      int(((actual_age >= 20) & (actual_age < 200)).sum()),
                      int(((actual_age >= 200) & (actual_age < 1000)).sum()),
                      int((actual_age >= 1000).sum())])
    if runner.logger.log_dir is not None:
        torch.save(dict(qpos=raw.sim.data.qpos.cpu().clone(), qvel=raw.sim.data.qvel.cpu().clone(),
                        age=actual_age.cpu(), target_age=age.cpu(),
                        observations={k: v.cpu().clone() for k, v in runner.env.get_observations().items()}),
                   Path(runner.logger.log_dir)/'phase-mixing-start.pt')
    runner._phase_mixing_ready = True
    return stats
