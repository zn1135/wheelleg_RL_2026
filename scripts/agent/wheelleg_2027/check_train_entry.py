"""Exercise the actual CLI and source snapshot with PPO learning intercepted."""
import argparse
import importlib
import json
from pathlib import Path
import sys

from ct_mjlab.scripts.train import main as train_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = importlib.import_module("ct_mjlab.envs.2027_wheelleg.runner")
    cls = module.WheellegRunner
    original = cls.learn_until

    def intercept_learning(self, deadline, max_iterations=None):
        assert self.env.task_name == "2027_wheelleg_walk"
        assert self.primary_updates == self.encoder_updates == 0
        assert self.iteration == 0 and max_iterations == 1
        model = self.env.env.sim.mj_model
        assert model.neq == 6
        hashes = json.loads((self.log_dir / "source_sha256.json").read_text())
        assert "resources/robots/2027_wheelleg/mjlab/wheelleg_2027.xml" in hashes
        assert "resources/robots/2027_wheelleg/meshes/base_link.STL" in hashes
        return dict(passed=True, task=self.env.task_name, optimizer_updates=0,
                    learning_intercepted=True, cli_entry_verified=True,
                    frozen_new_robot_assets=True)

    old_argv = sys.argv
    cls.learn_until = intercept_learning
    try:
        sys.argv = ["mjlab-train-check", "--task", "2027_wheelleg_walk",
                    "--num-envs", "20", "--device", "cuda:0", "--iterations", "1",
                    "--precheck", str(args.precheck), "--log-dir", str(args.output)]
        result = train_main()
        assert result["learning_intercepted"]
        print(json.dumps(result, indent=2))
    finally:
        cls.learn_until = original
        sys.argv = old_argv


if __name__ == "__main__":
    main()
