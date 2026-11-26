"""PPO runner with an independent 2027 robot/checkpoint identity."""
import hashlib
import torch

from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner
from .config import TASK
from .model import MODEL_XML, SPRING_FORCE_N

CONTRACT = "2027-wheelleg-walk-physical25-history125-v1"


def robot_record():
    return dict(
        robot="2027_wheelleg", model_sha256=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest(),
        asset_sha256={str(path.relative_to(MODEL_XML.parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(MODEL_XML.parents[1].rglob("*")) if path.is_file() and path.suffix.lower() in (".stl", ".xml", ".urdf")},
        wheel_reduction=15.7, spring_force_N=[SPRING_FORCE_N, SPRING_FORCE_N],
        spring_model="nominal-constant-push-unmeasured-curve",
    )


class WheellegRunner(FudanRunner):
    contract = CONTRACT

    def checkpoint_metadata(self):
        return dict(**super().checkpoint_metadata(), task=TASK, robot=robot_record())

    def load(self, path, **kwargs):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("task") != TASK or saved.get("robot") != robot_record():
            raise ValueError("2027 resume requires a checkpoint of this robot and physical asset")
        return super().load(path, **kwargs)
