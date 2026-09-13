"""CPU 检查首次落地门控：无策略阶段不运行网络，也不进入 PPO/encoder batch。"""

from unittest.mock import patch

import isaacgym  # 必须先于 torch
import torch

from wheel_legged_gym.rsl_rl.algorithms.ppo import PPO
from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence


def main():
    torch.manual_seed(0)
    actor_critic = ActorCriticSequence(
        num_obs=3,
        num_critic_obs=7,  # raw critic 4 + latent 3
        num_actions=2,
        num_encoder_obs=6,
        latent_dim=3,
        encoder_hidden_dims=[8],
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
    )
    ppo = PPO(
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        extra_learning_rate=1e-3,
    )
    ppo.init_storage(
        num_envs=3,
        num_transitions_per_env=2,
        actor_obs_shape=[3],
        critic_obs_shape=[7],
        obs_history_shape=[6],
        action_shape=[2],
    )
    obs = torch.randn(3, 3)
    obs_history = torch.randn(3, 6)
    critic_obs = torch.randn(3, 4)
    first_mask = torch.tensor([False, True, False])

    with torch.inference_mode():
        with patch.object(actor_critic, "act", wraps=actor_critic.act) as actor_call:
            actions = ppo.act(obs, obs_history, critic_obs, first_mask)
            assert actor_call.call_count == 1
            assert actor_call.call_args.args[0].shape[0] == 1
    assert torch.equal(actions[~first_mask], torch.zeros(2, 2))
    assert torch.equal(ppo.transition.valid_mask, first_mask)
    with torch.inference_mode():
        ppo.process_env_step(torch.ones(3), torch.zeros(3), {}, obs)

    no_policy_mask = torch.zeros(3, dtype=torch.bool)
    with torch.inference_mode():
        with patch.object(actor_critic, "act", wraps=actor_critic.act) as actor_call:
            actions = ppo.act(obs, obs_history, critic_obs, no_policy_mask)
            assert actor_call.call_count == 0
    assert torch.equal(actions, torch.zeros_like(actions))
    with torch.inference_mode():
        ppo.process_env_step(torch.zeros(3), torch.zeros(3), {}, obs)

    with torch.inference_mode():
        with patch.object(actor_critic, "encode", wraps=actor_critic.encode) as encoder_call:
            ppo.compute_returns(critic_obs, obs_history, no_policy_mask)
            assert encoder_call.call_count == 0
    assert ppo.storage.valid_masks.sum().item() == 1
    assert torch.isfinite(ppo.storage.advantages).all()
    losses = ppo.update()
    assert all(torch.isfinite(torch.tensor(losses)))
    print("PASS: inactive environments skip actor/critic/encoder inference")
    print("PASS: inactive free-fall steps are excluded from PPO and encoder batches")


if __name__ == "__main__":
    main()
