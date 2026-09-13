"""CPU 检查 checkpoint 续训状态；不创建仿真，不读取或改写历史 logs。"""

import tempfile
from pathlib import Path

import isaacgym  # 必须先于 torch
import torch
from torch import nn

from wheel_legged_gym.rsl_rl.algorithms.ppo import PPO
from wheel_legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner


class ProbePolicy(nn.Module):
    """仅提供 PPO 优化器所需的参数组，不运行策略或替代真实回放。"""

    def __init__(self, sequence=True):
        super().__init__()
        self.is_sequence = sequence
        self.actor = nn.Linear(2, 2)
        self.critic = nn.Linear(2, 1)
        self.std = nn.Parameter(torch.ones(2))
        if sequence:
            self.encoder = nn.Linear(2, 2)


def make_runner(lr=3e-4, sequence=True):
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.alg = PPO(
        ProbePolicy(sequence), learning_rate=lr, extra_learning_rate=lr,
        schedule="adaptive", device="cpu",
    )
    runner.current_learning_iteration = 0
    return runner


def populate_adam(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def main():
    torch.manual_seed(1)
    with tempfile.TemporaryDirectory(prefix="T-20260831-02-resume-check-") as tmp:
        modern_path = str(Path(tmp) / "modern.pt")
        legacy_path = str(Path(tmp) / "legacy.pt")
        source = make_runner(lr=1.5e-5)
        source.current_learning_iteration = 3000
        populate_adam(source.alg.optimizer)
        populate_adam(source.alg.extra_optimizer)
        source.save(modern_path, infos={"probe": True})

        resumed = make_runner(lr=1e-3)
        assert resumed.load(modern_path) == {"probe": True}
        assert resumed.current_learning_iteration == 3000
        assert resumed.alg.learning_rate == 1.5e-5
        assert_same(source.alg.optimizer.state_dict(), resumed.alg.optimizer.state_dict())
        assert_same(
            source.alg.extra_optimizer.state_dict(),
            resumed.alg.extra_optimizer.state_dict(),
        )
        assert_same(source.alg.actor_critic.state_dict(), resumed.alg.actor_critic.state_dict())
        print("PASS: network, iteration, PPO lr/Adam and encoder Adam round trip")

        legacy = torch.load(modern_path, map_location="cpu")
        legacy.pop("extra_optimizer_state_dict")
        torch.save(legacy, legacy_path)
        resumed_legacy = make_runner(lr=1e-3)
        resumed_legacy.load(legacy_path)
        assert resumed_legacy.alg.learning_rate == 1.5e-5
        assert resumed_legacy.alg.extra_optimizer.state_dict()["state"] == {}
        assert resumed_legacy.alg.extra_optimizer.param_groups[0]["lr"] == 1e-3
        print("PASS: old checkpoint keeps a fresh encoder optimizer and restores PPO lr")

        weights_only = make_runner(lr=3e-4)
        weights_only.load(modern_path, load_optimizer=False)
        assert weights_only.alg.learning_rate == 3e-4
        assert weights_only.alg.optimizer.state_dict()["state"] == {}
        assert weights_only.alg.extra_optimizer.state_dict()["state"] == {}
        assert_same(source.alg.actor_critic.state_dict(), weights_only.alg.actor_critic.state_dict())
        print("PASS: load_optimizer=False does not restore either optimizer")

        plain = make_runner(sequence=False)
        plain.save(modern_path)
        plain_loaded = make_runner(sequence=False)
        plain_loaded.load(modern_path)
        assert plain_loaded.alg.extra_optimizer is None
        print("PASS: non-sequence policy needs no encoder optimizer")

    print("CPU checks only; training, Isaac playback and MuJoCo behavior remain separate.")


if __name__ == "__main__":
    main()
