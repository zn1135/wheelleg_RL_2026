"""Scratch-only profile and explicit, same-task checkpoint continuation."""

import torch
from mjlab.utils.torch import configure_torch_backends

from .observations import OBSERVATION_CONTRACT
from ..chuanliantui.runner import DirectMotorOnPolicyRunner
from .mdp import CURRICULUM_VERSION, LEGACY_CURRICULUM_VERSION, STEPS_PER_ITERATION
from .rewards import RECIPE_VERSION


EXPERIMENT_PROFILE_CONTRACT = "chuanliantui-stand-39-82-v1"


class StandOnPolicyRunner(DirectMotorOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if train_cfg["num_steps_per_env"] != STEPS_PER_ITERATION:
            raise ValueError("Standing curriculum requires 24 rollout steps per iteration")
        for name in ("actor", "critic"):
            if train_cfg[name].get("obs_normalization", False):
                raise ValueError("Standing task retains fixed-scale observations")
        raw = env.unwrapped
        if abs(raw.physics_dt - .002) > 1e-10 or abs(raw.step_dt - .01) > 1e-10:
            raise ValueError("Standing task requires 500 Hz physics / 100 Hz policy")
        # Tiny action standard deviations must not amplify batch-dependent TF32 rounding.
        configure_torch_backends(allow_tf32=False)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # DirectMotor has no legacy INIT_CHECKPOINT environment-variable path.
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def save(self, path, infos=None):
        env = self.env.unwrapped
        state = env.command_manager.get_term("standup")
        return super().save(path, {
            **(infos or {}), "observation_contract": OBSERVATION_CONTRACT,
            "experiment_profile_contract": EXPERIMENT_PROFILE_CONTRACT,
            "physics_dt_s": env.physics_dt, "policy_dt_s": env.step_dt,
            "curriculum_version": CURRICULUM_VERSION,
            "stand_curriculum_stage": state.curriculum_stage,
            "recipe_version": RECIPE_VERSION,
            "stand_training_iteration": state.effective_iteration,
            "network_matmul_precision": torch.backends.cuda.matmul.fp32_precision,
            "training_randomization_scale": env.cfg.actions["pd"].randomization_scale,
            "observation_noise_scale": env.cfg.observations["actor"].terms["proprioception"].params["cfg"].noise_scale,
        })

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        required = {"observation_contract": OBSERVATION_CONTRACT,
                    "experiment_profile_contract": EXPERIMENT_PROFILE_CONTRACT,
                    "physics_dt_s": .002, "policy_dt_s": .01}
        for key, value in required.items():
            if infos.get(key) != value:
                raise ValueError(f"Incompatible standing {key}: {infos.get(key)!r}, expected {value!r}")
        version = infos.get("curriculum_version")
        if version not in (CURRICULUM_VERSION, LEGACY_CURRICULUM_VERSION):
            raise ValueError(f"Incompatible standing curriculum version: {version!r}")
        saved_stage = infos.get("stand_curriculum_stage") if version == CURRICULUM_VERSION else 0
        if type(saved_stage) is not int or saved_stage not in (0, 1, 2):
            raise ValueError(f"Invalid standing curriculum stage: {saved_stage!r}")
        iteration = infos.get("stand_training_iteration")
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Missing or invalid standing curriculum iteration")
        del checkpoint
        restored = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        env = self.env.unwrapped
        state = env.command_manager.get_term("standup")
        configured_stage = state.cfg.curriculum_stage
        state.curriculum_stage = saved_stage if configured_stage is None else configured_stage
        print(f"[INFO] Standing curriculum: checkpoint {version} stage={saved_stage}; "
              f"configured={configured_stage}; active={state.curriculum_stage}", flush=True)
        state.training_iteration = iteration - env.common_step_counter // STEPS_PER_ITERATION
        if load_cfg is None or load_cfg.get("iteration", False):
            # RSL saves the last zero-based index; resume at the next completed update.
            self.current_learning_iteration = iteration
        state.assign_disturbances(torch.arange(env.num_envs, device=env.device))
        return restored
