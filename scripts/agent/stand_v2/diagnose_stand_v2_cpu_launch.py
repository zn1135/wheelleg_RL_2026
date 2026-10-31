#!/usr/bin/env python3
"""Trace the reported native CPU launch, with fixed initial state and no learning."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"sim2sim"))
from mj_sim2sim_chuanliantui_stand_v2 import StandV2Playback, OnnxPolicy, DT
from mj_sim2sim_huananhu_multimode import sha256, MODEL_ASSET


def measure(args, variant):
    env = StandV2Playback(args.initial_states, args.sample, integration_substeps=1)
    policy = OnnxPolicy(args.checkpoint, args.onnx)
    substeps = 4 if variant == "substeps4" else 1
    env.model.opt.timestep = DT/substeps
    original_controls = env.controls

    def controls(action):
        torque, spring = original_controls(action)
        if variant in ("motors_off", "all_off"):
            torque[:] = 0.
        if variant == "all_off":
            spring[:] = 0.
        return torque, spring

    env.controls = controls
    rows, qpos, qvel, force_constraint, force_actuator = [], [], [], [], []
    work = 0.
    original_step = mujoco.mj_step

    def step(model, data):
        nonlocal work
        for _ in range(substeps):
            before_v = data.qvel.copy()
            original_step(model, data)
            work += float(data.qfrc_actuator @ ((before_v+data.qvel)*.5))*model.opt.timestep
            mujoco.mj_energyPos(model, data)
            mujoco.mj_energyVel(model, data)
            rows.append([data.time, data.qpos[2], data.qvel[2], *data.energy, work,
                np.max(np.abs(data.qfrc_constraint)), np.max(np.abs(data.qfrc_actuator)),
                min(data.contact.dist, default=0.), data.ncon])
            qpos.append(data.qpos.copy()); qvel.append(data.qvel.copy())
            force_constraint.append(data.qfrc_constraint.copy())
            force_actuator.append(data.qfrc_actuator.copy())

    mujoco.mj_step = step
    try:
        for _ in range(round(args.seconds/.01)):
            env.step(policy)
    finally:
        mujoco.mj_step = original_step
    trace = np.asarray(rows)
    np.savez_compressed(args.output/f"{variant}.npz", trace=trace, qpos=qpos, qvel=qvel,
        force_constraint=force_constraint, force_actuator=force_actuator,
        fields=["time", "height", "world_vz", "potential", "kinetic", "actuator_work",
                "max_constraint_force", "max_actuator_force", "min_contact_distance", "contacts"])
    peak = int(trace[:, 1].argmax())
    result = dict(variant=variant, peak_height_m=float(trace[peak, 1]), peak_time_s=float(trace[peak, 0]),
        max_upward_speed_m_s=float(trace[:, 2].max()), max_kinetic_J=float(trace[:, 4].max()),
        max_actuator_work_J=float(trace[:, 5].max()),
        max_constraint_force=float(trace[:, 6].max()), deepest_contact_m=float(trace[:, 8].min()),
        max_mechanical_energy_J=float(trace[:, 3:5].sum(-1).max()),
        initial_mechanical_energy_J=float(trace[0, 3:5].sum()),
        samples=len(trace), physics_dt_s=env.model.opt.timestep,
        parameter_updates=0, has_stood=bool(env.has_stood))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--initial-states", type=Path, required=True)
    p.add_argument("--sample", type=int, default=17)
    p.add_argument("--seconds", type=float, default=3.)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = [measure(args, variant) for variant in ("baseline", "motors_off", "all_off", "substeps4")]
    (args.output/"result.json").write_text(json.dumps(dict(results=results, sample=args.sample,
        checkpoint_sha256=sha256(args.checkpoint), xml_sha256=sha256(MODEL_ASSET),
        source_sha256=sha256(Path(__file__)), parameter_updates=0), indent=2)+"\n")
