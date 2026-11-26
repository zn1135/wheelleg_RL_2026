"""Check retained imports, syntax and rejection before any training update."""
from pathlib import Path
import ast
import importlib
import json
import subprocess
from types import SimpleNamespace

root=Path.cwd()
modules=("ct_mjlab.supermimic.__main__","ct_mjlab.smp.ppo","ct_mjlab.bc",
         "ct_mjlab.envs.chuanliantui.fudan_ppo_step.train",
         "ct_mjlab.envs.chuanliantui.fudan_ppo_jump.train",
         "ct_mjlab.envs.2027_wheelleg.train","ct_mjlab.envs.2027_wheelleg.play")
for name in modules:importlib.import_module(name)
changed=subprocess.check_output(["git","diff","--name-only"],text=True).splitlines()
files=[root/p for p in changed if p.endswith(".py") and (root/p).is_file()]
files+=list((root/"mjlab_training/src/ct_mjlab/envs/2027_wheelleg").glob("*.py"))
files+=list((root/"mjlab_training/src/ct_mjlab/envs/chuanliantui").rglob("*.py"))
files+=list((root/"scripts/agent/wheelleg_2027").glob("*.py"))
for path in set(files):
 compile(path.read_text(),str(path),"exec")
from ct_mjlab.scripts.train import train
try:
 train(SimpleNamespace(task="2027_wheelleg_walk",num_envs=20,log_dir=Path("/tmp/not-created-2027"),deadline=None,iterations=None))
except ValueError as error:
 assert "budget" in str(error)
else:raise AssertionError("Training started without a budget")
assert not Path("/tmp/not-created-2027").exists()
runner_module=importlib.import_module("ct_mjlab.envs.2027_wheelleg.runner")
runner=runner_module.WheellegRunner.__new__(runner_module.WheellegRunner)
try:
 runner.load(root/"logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt")
except ValueError as error:
 assert "2027" in str(error)
else:raise AssertionError("2027 accepted the old robot checkpoint")
print(json.dumps(dict(imports=len(modules),compiled_files=len(set(files)),
                       unauthorized_training_budget_rejected=True,cross_robot_resume_rejected=True)))
