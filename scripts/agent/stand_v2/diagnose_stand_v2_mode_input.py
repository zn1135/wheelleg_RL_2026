#!/usr/bin/env python3
"""Ablate only the policy-visible NORMAL/RECOVER flag of a frozen stand_v2 model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand_v2.observations import get_state
from scripts.agent.stand_v2.assess_chuanliantui_stand_v2 import main as assess, digest


def run(args):
    original_make = task_registry.make_env
    audits = []

    def make_env(*positional, **keywords):
        wrapped, cfg = original_make(*positional, **keywords)
        raw = wrapped.unwrapped
        observations = get_state(raw)
        original_mode = observations.mode

        def mode():
            values = original_mode()
            if args.mode == "baseline":
                return values
            changed = values.clone()
            changed[:, :5] = 0.
            changed[:, 0 if args.mode == "normal" else 3] = 1.
            return changed

        observations.mode = mode
        step = wrapped.step
        actual, seen = [], []
        audit = dict(steps=0, non_mode_input_max_difference=0.)
        audits.append((audit, actual, seen))

        def audited_step(action):
            obs = wrapped.get_observations()
            a, h = obs["actor"], obs["history"].reshape(raw.num_envs, 5, 32)
            assert torch.equal(a, h[:, -1])
            assert torch.equal(a[:, 25:32], mode())
            if args.mode != "baseline":
                index = 0 if args.mode == "normal" else 3
                assert (h[:, :, 25:30].argmax(-1) == index).all()
                assert (h[:, :, 25:30].sum(-1) == 1.).all()
            if not audit["steps"]:
                # Same state, two observation reads: only the five one-hot channels may change.
                observations.mode = original_mode
                untouched = observations.actor()
                observations.mode = mode
                changed = observations.actor()
                keep = list(range(25))+[30, 31]
                error = (untouched[:, keep]-changed[:, keep]).abs().max()
                assert error == 0.
                audit["non_mode_input_max_difference"] = float(error)
            actual.append(original_mode()[:, :5].argmax(-1).clone())
            seen.append(a[:, 25:30].argmax(-1).clone())
            audit["steps"] += 1
            return step(action)

        wrapped.step = audited_step
        return wrapped, cfg

    task_registry.make_env = make_env
    try:
        assess(argparse.Namespace(output=args.output, checkpoint=args.checkpoint,
            num_envs=64, seed=42, device="cuda:0"))
        audit, actual, seen = audits[0]
        assert audit["steps"] == 2000
        np.savez_compressed(args.output/"mode-input.npz",
            actual_mode=torch.stack(actual).cpu().numpy(),
            actor_mode=torch.stack(seen).cpu().numpy())
        result = dict(mode=args.mode, **audit, seed=42, num_envs=64, duration_s=20.,
            model_unchanged=True, physical_state_machine_unchanged=True,
            reward_and_termination_unchanged=True, actor_current_and_all_history_checked=True,
            source_sha256=digest(__file__), mode_trace_sha256=digest(args.output/"mode-input.npz"),
            scope="Mode input ablation from reset, including history and encoder inputs. "
                  "Forced flags can be outside the training distribution; not a new production rule.")
        (args.output/"mode-ablation.json").write_text(json.dumps(result, indent=2)+"\n")
        (args.output/"mode-source.py").write_bytes(Path(__file__).read_bytes())
    finally:
        task_registry.make_env = original_make


def compare(args):
    names = ("baseline", "baseline_repeat", "normal", "recover")
    reports, states = {}, {}
    for name in names:
        p = args.root/name
        r = json.loads((p/"result.json").read_text())
        audit = json.loads((p/"mode-ablation.json").read_text())
        assert r["parameter_updates"] == 0 and r["actor_and_encoder_unchanged"]
        assert audit["steps"] == 2000 and audit["non_mode_input_max_difference"] == 0.
        assert digest(p/"physical.npz") == r["physical_sha256"]
        assert digest(p/"qpos.npz") == r["qpos_sha256"]
        assert digest(p/"mode-input.npz") == audit["mode_trace_sha256"]
        with np.load(p/"physical.npz") as data:
            states[name] = {k:data[k].copy() for k in ("initial_qpos", "initial_qvel", "weight_N")}
        with np.load(p/"mode-input.npz") as data:
            seen = data["actor_mode"]
            actual = data["actual_mode"]
            if name.startswith("baseline"):
                assert np.array_equal(seen, actual)
            else:
                assert (seen == (0 if name == "normal" else 3)).all()
        reports[name] = r
    for name in names:
        for key in ("checkpoint_sha256", "xml_sha256", "criterion", "num_envs", "duration_s", "seed"):
            assert reports[name][key] == reports["baseline"][key], (name, key)
        for key in states[name]:
            assert np.array_equal(states[name][key], states["baseline"][key]), (name, key)
    successes = {name:{row["env"] for row in r["rows"] if row["passed"]} for name,r in reports.items()}
    groups = {}
    for name in names:
        r = reports[name]
        groups[name] = dict(passed=r["passed"], ever_stood=r["ever_stood"],
            ever_inverted_envs=r["ever_inverted_envs"],
            failed_envs=sorted(set(range(64))-successes[name]),
            lost_vs_baseline=sorted(successes["baseline"]-successes[name]),
            gained_vs_baseline=sorted(successes[name]-successes["baseline"]))
    result = dict(checkpoint_sha256=reports["baseline"]["checkpoint_sha256"],
        initial_states_and_weight_exact_match=True, mode_inputs_verified=True,
        model_unchanged=True, parameter_updates=0, groups=groups,
        scope="Single seed, same frozen model, 64 initial states, 20s/500Hz final balance. "
              "Baseline repeat measures run variability. This does not identify the training-time cause of regression.")
    with (args.root/"comparison.json").open("x") as output:
        output.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("run")
    measure.add_argument("--checkpoint", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--mode", choices=("baseline", "normal", "recover"), required=True)
    summary = commands.add_parser("compare")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args) if args.command == "run" else compare(args)
