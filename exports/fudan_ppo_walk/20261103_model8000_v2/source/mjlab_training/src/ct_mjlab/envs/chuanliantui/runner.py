"""Prevent old virtual-joint policies from driving physical motor coordinates."""

import torch

from mjlab.rl import MjlabOnPolicyRunner

from .pd_action import CONTROL_CONTRACT


class DirectMotorOnPolicyRunner(MjlabOnPolicyRunner):
    def save(self, path: str, infos=None) -> None:
        super().save(path, {**(infos or {}), "control_contract": CONTROL_CONTRACT})

    def load(self, path: str, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        contract = (checkpoint.get("infos") or {}).get("control_contract")
        if contract != CONTROL_CONTRACT:
            raise ValueError(
                f"Incompatible chuanliantui control contract: {contract!r}. "
                "This task uses direct physical motor PD, physical gas springs "
                "and raw policy actions with bounded physical targets; "
                "incompatible checkpoints require retraining."
            )
        del checkpoint
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
