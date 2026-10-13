"""Checkpoint contracts and iteration continuity for the flat Huananhu task."""

import hashlib
import os

import torch

from .huananhu_observations import OBSERVATION_CONTRACT
from .runner import DirectMotorOnPolicyRunner


EXPERIMENT_PROFILE_CONTRACT = "chuanliantui-huananhu-flat-35-78-ppo-sourcexml-spring90mm-velocity-reciprocal-no-knee-limit-wide-height-gate-mixed-reset-velocity-heading-r14"
STEPS_PER_ITERATION = 24
# Fine-tuning entry: previous experiment profiles whose weights may initialize
# this one. They must share the control and observation contracts (still
# checked); only the reward/command profile differs. Set via environment so
# the generic mjlab train CLI needs no extra flag.
INIT_CHECKPOINT_ENV = "WHEELLEG_HUANANHU_INIT_CHECKPOINT"
INIT_COMPATIBLE_PROFILES = (
    "chuanliantui-huananhu-flat-35-78-ppo-sourcexml-spring90mm-v7init-gate3-6cm-kneelimit-h27-34-v11",
)


class HuananhuOnPolicyRunner(DirectMotorOnPolicyRunner):
    """Keep fixed-scale observations and command curriculum across resumes."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        for name in ("actor", "critic"):
            if train_cfg[name].get("obs_normalization", False):
                raise ValueError(
                    f"Huananhu fixed-scale observations require {name}.obs_normalization=False"
                )
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        init_path = os.environ.get(INIT_CHECKPOINT_ENV)
        if init_path and self.current_learning_iteration == 0 and not hasattr(self, "init_source"):
            self.initialize_from(init_path)
        # The generic CLI randomizes episode ages after reset. That skips the
        # first ground drop's 0.2 s torque gate and 1.5 s command override.
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def initialize_from(self, path: str) -> None:
        """Load actor, critic and optimizer from a compatible earlier profile.

        Iteration numbering restarts at 0 for this experiment while the command
        curriculum continues from the source checkpoint's effective iteration.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        profile = infos.get("experiment_profile_contract")
        if profile not in INIT_COMPATIBLE_PROFILES:
            raise ValueError(
                f"Huananhu init checkpoint profile {profile!r} is not in {INIT_COMPATIBLE_PROFILES!r}")
        if infos.get("observation_contract") != OBSERVATION_CONTRACT:
            raise ValueError(
                f"Incompatible Huananhu observation contract in init checkpoint: "
                f"{infos.get('observation_contract')!r}; expected {OBSERVATION_CONTRACT!r}")
        if "huananhu_training_iteration" not in infos:
            raise ValueError("Huananhu init checkpoint is missing curriculum training iteration")
        with open(path, "rb") as file:
            digest = hashlib.sha256(file.read()).hexdigest()
        effective_iteration = int(infos["huananhu_training_iteration"])
        del checkpoint
        # Control contract is enforced by DirectMotorOnPolicyRunner.load.
        DirectMotorOnPolicyRunner.load(
            self, path, load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": False},
            strict=True)
        env = self.env.unwrapped
        owner = self._progress_owner()
        offset = effective_iteration - int(env.common_step_counter) // STEPS_PER_ITERATION
        if owner is env:
            env.huananhu_training_iteration = offset
        else:
            owner.training_iteration = offset
        self.init_source = {"path": os.path.abspath(path), "sha256": digest,
                            "experiment_profile_contract": profile,
                            "huananhu_training_iteration": effective_iteration}
        print(f"[INFO] Huananhu init from {path} (profile {profile}, curriculum iteration {effective_iteration})")

    def _progress_owner(self):
        env = self.env.unwrapped
        command = env.command_manager.get_term("standup")
        return command if hasattr(command, "training_iteration") else env

    def _effective_training_iteration(self) -> int:
        env = self.env.unwrapped
        owner = self._progress_owner()
        offset = int(getattr(owner, "training_iteration", getattr(owner, "huananhu_training_iteration", 0)))
        return offset + int(env.common_step_counter) // STEPS_PER_ITERATION

    def _training_stage(self) -> dict:
        cfg = getattr(self.env.unwrapped.command_manager.get_term('standup'), 'cfg', None)
        defaults = {'training_stage': 'v12_mixed', 'ground_start_probability': .5,
                    'zero_velocity_commands': False, 'height_range': (.27, .34),
                    'strict_recovery_metrics': False, 'ground_randomization_probability': 0.,
                    'ground_tilt_max_rad': 0., 'ground_yaw_max_rad': 0.,
                    'ground_height_offset_range': (0., 0.)}
        from .huananhu_env_cfg import COMMAND_CONFIG_FIELDS
        defaults.update({name: getattr(cfg, name) for name in COMMAND_CONFIG_FIELDS})
        return {name: getattr(cfg, name, value) for name, value in defaults.items()}

    def save(self, path: str, infos=None) -> None:
        super().save(path, {
            **(infos or {}),
            "observation_contract": OBSERVATION_CONTRACT,
            "experiment_profile_contract": EXPERIMENT_PROFILE_CONTRACT,
            "huananhu_training_iteration": self._effective_training_iteration(),
            "training_line": "recover",
            "training_stage_config": self._training_stage(),
            **({"init_source": self.init_source} if hasattr(self, "init_source") else {}),
            **({"stage_history": self.stage_history} if hasattr(self, "stage_history") else {}),
        })

    def load(self, path: str, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        if infos.get("observation_contract") != OBSERVATION_CONTRACT:
            raise ValueError(
                f"Incompatible Huananhu observation contract: {infos.get('observation_contract')!r}; "
                f"expected {OBSERVATION_CONTRACT!r}"
            )
        if infos.get("experiment_profile_contract") != EXPERIMENT_PROFILE_CONTRACT:
            raise ValueError(
                f"Incompatible Huananhu experiment profile: {infos.get('experiment_profile_contract')!r}; "
                f"expected {EXPERIMENT_PROFILE_CONTRACT!r}"
            )
        if "huananhu_training_iteration" not in infos:
            raise ValueError("Huananhu checkpoint is missing curriculum training iteration")
        effective_iteration = int(infos["huananhu_training_iteration"])
        if effective_iteration < 0:
            raise ValueError("Huananhu checkpoint has a negative curriculum training iteration")
        del checkpoint

        restored_infos = super().load(
            path, load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        env = self.env.unwrapped
        owner = self._progress_owner()
        offset = effective_iteration - int(env.common_step_counter) // STEPS_PER_ITERATION
        if owner is env:
            env.huananhu_training_iteration = offset
        else:
            owner.training_iteration = offset
        if infos.get('init_source') is not None:
            self.init_source = infos['init_source']
        self.stage_history = list(infos.get('stage_history', []))
        previous_stage = infos.get('training_stage_config', {
            'training_stage': 'v12_mixed', 'ground_start_probability': .5,
            'zero_velocity_commands': False, 'height_range': (.27, .34),
            'strict_recovery_metrics': False})
        current_stage = self._training_stage()
        # Older stages predate configurable root-pose perturbations. Fill their
        # zero defaults without creating a spurious transition on resume.
        previous_stage = {'ground_randomization_probability': 0., 'ground_tilt_max_rad': 0.,
                          'ground_yaw_max_rad': 0., 'ground_height_offset_range': (0., 0.),
                          **previous_stage}
        if previous_stage != current_stage:
            with open(path, 'rb') as file:
                digest = hashlib.sha256(file.read()).hexdigest()
            self.stage_history.append({'from': previous_stage, 'to': current_stage,
                                       'source_checkpoint': os.path.abspath(path), 'sha256': digest,
                                       'huananhu_training_iteration': effective_iteration})
        return restored_infos
