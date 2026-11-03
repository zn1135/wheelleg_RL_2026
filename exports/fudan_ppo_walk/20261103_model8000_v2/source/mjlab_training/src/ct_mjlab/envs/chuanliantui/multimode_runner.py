"""Strict multimode checkpoints and explicit v12 policy/optimizer migration."""

import hashlib
import os

import torch

from .huananhu_observations import OBSERVATION_CONTRACT as FLAT_OBSERVATION_CONTRACT
from .huananhu_runner import (EXPERIMENT_PROFILE_CONTRACT as FLAT_EXPERIMENT_CONTRACT,
                              HuananhuOnPolicyRunner, STEPS_PER_ITERATION)
from .multimode_observations import OBSERVATION_CONTRACT
from .pd_action import CONTROL_CONTRACT
from .runner import DirectMotorOnPolicyRunner


EXPERIMENT_PROFILE_CONTRACT = "chuanliantui-huananhu-multimode-39-82-v1"
INIT_CHECKPOINT_ENV = "WHEELLEG_MULTIMODE_INIT_CHECKPOINT"


def migrated_model_state(old: dict, new: dict, *, old_dim: int, mode_start: int) -> dict:
    """Keep the old function independent of newly active mode inputs at transfer.

    Previously normal=1 and all other modes=0, so its contribution can be
    absorbed exactly into the first-layer bias. All new mode and appended flag
    columns start at zero. Remaining weights, including actor std, are copied.
    """
    result = {key: value.clone() for key, value in new.items()}
    for key, value in old.items():
        if key not in result:
            raise ValueError(f"Missing destination model parameter: {key}")
        if key == "mlp.0.weight":
            if value.shape[1] != old_dim or result[key].shape[1] != old_dim + 4:
                raise ValueError("Unexpected first-layer input dimensions during v12 migration")
            result[key].zero_()
            result[key][:, :old_dim].copy_(value)
            result[key][:, mode_start:mode_start + 7].zero_()
        elif key == "mlp.0.bias":
            result[key].copy_(value + old["mlp.0.weight"][:, mode_start])
        else:
            if result[key].shape != value.shape:
                raise ValueError(f"Unexpected parameter shape: {key}")
            result[key].copy_(value)
    if set(result) != set(old):
        raise ValueError("Source and destination model parameter sets differ")
    return result


def compatible_optimizer_state(old: dict, actor, critic) -> dict:
    """Retain Adam history for unchanged layers; reset both input-layer moments."""
    result = {"state": {key: dict(value) for key, value in old["state"].items()},
              "param_groups": [dict(group) for group in old["param_groups"]]}
    parameters = list(actor.named_parameters()) + list(critic.named_parameters())
    ids = [key for group in result["param_groups"] for key in group["params"]]
    if len(ids) != len(parameters):
        raise ValueError("Optimizer parameter count differs from v12")
    for key, (name, _param) in zip(ids, parameters, strict=True):
        if name in ("mlp.0.weight", "mlp.0.bias"):
            result["state"].pop(key, None)
    return result


class MultimodeOnPolicyRunner(HuananhuOnPolicyRunner):
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        path = os.environ.get(INIT_CHECKPOINT_ENV)
        if path and self.current_learning_iteration == 0 and not hasattr(self, "init_source"):
            self.initialize_from(path)
        # Preserve the ground-start gate and use this task's explicit migration
        # path even if the flat task's initializer environment variable is set.
        return DirectMotorOnPolicyRunner.learn(
            self, num_learning_iterations, init_at_random_ep_len=False)

    def initialize_from(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        required = {"control_contract": CONTROL_CONTRACT,
                    "observation_contract": FLAT_OBSERVATION_CONTRACT,
                    "experiment_profile_contract": FLAT_EXPERIMENT_CONTRACT}
        for key, wanted in required.items():
            if infos.get(key) != wanted:
                raise ValueError(f"v12 migration {key}: {infos.get(key)!r}, expected {wanted!r}")
        iteration = int(infos["huananhu_training_iteration"])
        if iteration < 0:
            raise ValueError("Negative v12 curriculum iteration")
        actor = self.alg._raw_actor
        critic = self.alg._raw_critic
        actor.load_state_dict(migrated_model_state(
            checkpoint["actor_state_dict"], actor.state_dict(), old_dim=35, mode_start=28))
        critic.load_state_dict(migrated_model_state(
            checkpoint["critic_state_dict"], critic.state_dict(), old_dim=78, mode_start=32))
        optimizer = compatible_optimizer_state(checkpoint["optimizer_state_dict"], actor, critic)
        self.alg.optimizer.load_state_dict(optimizer)
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        env = self.env.unwrapped
        env.command_manager.get_term("standup").training_iteration = (
            iteration - int(env.common_step_counter) // STEPS_PER_ITERATION)
        with open(path, "rb") as file:
            digest = hashlib.sha256(file.read()).hexdigest()
        self.init_source = {"path": os.path.abspath(path), "sha256": digest,
                            "experiment_profile_contract": FLAT_EXPERIMENT_CONTRACT,
                            "huananhu_training_iteration": iteration,
                            "input_migration": "normal-column-to-bias, four zero columns; first-layer Adam reset"}
        print(f"[INFO] Multimode migrated from v12 {path} (curriculum {iteration})", flush=True)

    def save(self, path: str, infos=None) -> None:
        DirectMotorOnPolicyRunner.save(self, path, {
            **(infos or {}), "observation_contract": OBSERVATION_CONTRACT,
            "experiment_profile_contract": EXPERIMENT_PROFILE_CONTRACT,
            "huananhu_training_iteration": self._effective_training_iteration(),
            **({"init_source": self.init_source} if hasattr(self, "init_source") else {}),
        })

    def load(self, path: str, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        for key, wanted in (("observation_contract", OBSERVATION_CONTRACT),
                            ("experiment_profile_contract", EXPERIMENT_PROFILE_CONTRACT),
                            ("control_contract", CONTROL_CONTRACT)):
            if infos.get(key) != wanted:
                raise ValueError(f"Incompatible multimode {key}: {infos.get(key)!r}")
        if "huananhu_training_iteration" not in infos:
            raise ValueError("Missing multimode curriculum iteration")
        restored = DirectMotorOnPolicyRunner.load(
            self, path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        owner = self._progress_owner()
        owner.training_iteration = int(infos["huananhu_training_iteration"]) - (
            int(self.env.unwrapped.common_step_counter) // STEPS_PER_ITERATION)
        if infos.get("init_source") is not None:
            self.init_source = infos["init_source"]
        return restored
