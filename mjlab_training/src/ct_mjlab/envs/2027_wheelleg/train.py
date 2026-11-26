"""Bounded 2027 training through the standard mjlab task entry."""
import json
from pathlib import Path
import time

from ct_mjlab.utils.task_registry import task_registry
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import snapshot
from .config import TASK, config_record
from .runner import robot_record


def train(args):
    if args.num_envs is None or args.num_envs <= 0 or args.log_dir is None:
        raise ValueError("2027 training requires --num-envs and a new --log-dir")
    if (args.deadline is None) == (args.iterations is None):
        raise ValueError("Specify exactly one --iterations or --deadline budget")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("Iteration budget must be positive")
    if args.deadline is not None and args.deadline <= time.time():
        raise ValueError("The requested training deadline has expired")
    if not args.device.startswith("cuda"):
        raise ValueError("2027 training uses the GPU physical backend")
    if args.precheck is None:
        raise ValueError("Supply --precheck from scripts.agent.wheelleg_2027.check")
    proof = json.loads(args.precheck.read_text())
    if not proof.get("passed") or proof.get("task") != TASK or proof.get("robot") != robot_record():
        raise ValueError("The physical precheck does not match the current 2027 asset")
    cfg, ppo = task_registry.get_cfgs(TASK)
    if proof.get("configuration") != config_record(cfg):
        raise ValueError("The physical precheck configuration differs from training")
    if args.seed is not None:
        ppo.seed = args.seed
    output = Path(args.log_dir)
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        env, _ = task_registry.make_env(TASK, env_cfg=cfg, train_cfg=ppo,
            num_envs=args.num_envs, seed=ppo.seed, device=args.device)
        runner, _ = task_registry.make_alg_runner(env, TASK, train_cfg=ppo,
            log_dir=output, device=args.device, resume=args.resume)
        snapshot(output)
        asset_root = Path(__file__).resolve().parents[5] / "resources/robots/2027_wheelleg"
        import shutil
        shutil.copytree(asset_root, output / "source/resources/robots/2027_wheelleg")
        hashes_file = output / "source_sha256.json"
        hashes = json.loads(hashes_file.read_text())
        import hashlib
        for path in asset_root.rglob("*"):
            if path.is_file():
                hashes[str(path.relative_to(asset_root.parents[2]))] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes_file.write_text(json.dumps(hashes, indent=2) + "\n")
        report = runner.learn_until(float("inf"), max_iterations=runner.iteration + args.iterations) if args.iterations is not None else runner.learn_until(args.deadline)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if env is not None:
            env.close()
