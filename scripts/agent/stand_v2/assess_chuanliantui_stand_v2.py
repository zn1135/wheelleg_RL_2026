#!/usr/bin/env python3
"""20 s stand_v2 acceptance: 500 Hz balance, drift reported separately; no learning."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg

from ct_mjlab.envs import task_registry
from scripts.agent.stand_v2.diagnose_chuanliantui_stand_v2 import diagnose


CRITERION = "stand-v2-rise-balance-drift-separated-v1"
FIELDS = ("left_support_N", "right_support_N", "base_force_N", "roll_rad", "pitch_rad",
          "roll_pitch_speed_rad_s", "height_m", "vx_world_m_s", "vy_world_m_s",
          "x_m", "y_m", "max_leg_speed_rad_s", "finite", "active", "gravity_z")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_physics(env):
    recorder = getattr(env, "_stand_v2_acceptance", None)
    if recorder is not None:
        recorder.sample()
    return torch.zeros(env.num_envs, device=env.device)


class Recorder:
    def __init__(self, raw):
        self.raw, self.count = raw, 0
        robot = raw.scene["robot"]
        ids, _ = robot.find_joints(("lf0", "lf00", "rf0", "rf00"), preserve_order=True)
        self.leg_vadr = robot.indexing.joint_v_adr[ids]
        self.root_qadr = robot.indexing.free_joint_q_adr
        self.root_vadr = robot.indexing.free_joint_v_adr
        self.live = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.trace = torch.empty((round(20. / raw.physics_dt), raw.num_envs, len(FIELDS)),
                                 device=raw.device)
        self.qpos = [raw.sim.data.qpos.clone()]

    def sample(self):
        raw, d = self.raw, self.raw.sim.data
        root, velocity = d.qpos[:, self.root_qadr], d.qvel[:, self.root_vadr]
        w, x, y, z = root[:, 3:].unbind(-1)
        gx, gy, gz = 2*(w*y-x*z), -2*(y*z+w*x), -1+2*(x*x+y*y)
        roll = torch.atan2(-gy, -gz).abs()
        pitch = gx.clamp(-1., 1.).asin().abs()
        support = -raw.scene["wheel_contact"].data.force[:, :, 2]
        base = raw.scene["base_contact"].data.force.norm(dim=-1).amax(-1)
        finite = (torch.isfinite(d.qpos).all(-1) & torch.isfinite(d.qvel).all(-1)
                  & torch.isfinite(support).all(-1) & torch.isfinite(base))
        self.trace[self.count] = torch.stack((*support.unbind(-1), base, roll, pitch,
            velocity[:, 3:5].norm(dim=-1), root[:, 2]-raw.scene.env_origins[:, 2],
            velocity[:, 0], velocity[:, 1], root[:, 0], root[:, 1],
            d.qvel[:, self.leg_vadr].abs().amax(-1), finite.float(), self.live.float(), gz), -1)
        self.count += 1


def summarize(trace, weight, rollout, dt):
    """Use consecutive physics ticks, so a brief support loss resets the final hold."""
    wheels = trace[:, :, :2]
    conditions = {
        "each_wheel": (wheels >= np.maximum(.1*weight, 5.)[None, :, None]).all(-1),
        "wheel_total": ((wheels.sum(-1) >= .8*weight) & (wheels.sum(-1) <= 1.2*weight)),
        "body_clear": trace[:, :, 2] <= 1.,
        "tilt": trace[:, :, 3:5].max(-1) <= np.deg2rad(35.),
        "angular_speed": trace[:, :, 5] <= .5,
        "finite": trace[:, :, 12] > .5,
        "active": trace[:, :, 13] > .5,
    }
    balanced = np.logical_and.reduce(list(conditions.values()))
    hold = np.zeros(len(weight), dtype=np.int64)
    for sample in balanced:
        hold = np.where(sample, hold+1, 0)
    rows = []
    for i, old in enumerate(rollout["episodes"]):
        live = conditions["active"][:, i]
        finite = bool(conditions["finite"][live, i].all())
        full = bool(live.all() and old["survived_to_end"] and finite)
        final = bool(full and hold[i] >= round(1./dt))
        stood = bool(old["ever_isaac_stood"])
        v = trace[-round(5./dt):, i]
        speed = np.linalg.norm(v[:, 7:9], axis=-1)
        last_second = slice(-round(1./dt), None)
        reasons = [name for name, mask in conditions.items() if not mask[last_second, i].all()]
        if not stood:
            reasons.append("never_stood")
        if not old["survived_to_end"]:
            reasons.append("early_termination")
        rows.append(dict(env=i, ever_stood=stood, full_20s=full, finite=finite,
            final_balanced=final, final_balance_hold_s=float(hold[i]*dt),
            passed=bool(stood and final), failure_reasons=reasons,
            ever_inverted=bool(((trace[:, i, 14] > 0.) & live).any()),
            last5_median_speed_m_s=float(np.median(speed)), last5_max_speed_m_s=float(speed.max()),
            last5_xy_excursion_m=float(np.linalg.norm(v[:, 9:11]-v[0, 9:11], axis=-1).max()),
            last5_median_height_m=float(np.median(v[:, 6])),
            last5_median_tilt_deg=float(np.rad2deg(np.median(v[:, 3:5].max(-1)))),
            last5_median_base_force_N=float(np.median(v[:, 2])),
            last5_median_wheel_support_N=np.median(v[:, :2], axis=0).tolist(),
            final1_condition_fraction={name: float(mask[last_second, i].mean())
                                       for name, mask in conditions.items()}))
    return rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original_make_env = task_registry.make_env
    recorder = []
    frozen_actor = []
    original_make_runner = task_registry.make_alg_runner

    def make_env(name, **kwargs):
        kwargs["env_cfg"].metrics["stand_v2_acceptance"] = MetricsTermCfg(
            func=sample_physics, per_substep=True, reduce="last")
        wrapped, cfg = original_make_env(name, **kwargs)
        raw = wrapped.unwrapped
        assert raw.physics_dt == .002 and raw.step_dt == .01
        r = Recorder(raw)
        raw._stand_v2_acceptance = r
        recorder.append(r)
        mujoco.mj_saveModel(raw.sim.mj_model, str(args.output/"scene.mjb"), None)
        step = wrapped.step

        def recorded_step(action):
            if not frozen_actor:
                frozen_actor.append({k: v.clone() for k, v in runners[0].alg.actor.state_dict().items()})
            result = step(action)
            r.qpos.append(raw.sim.data.qpos.clone())
            # Explicit resets in the diagnostic must never make a failed case active again.
            r.live &= ~result[2].bool()
            return result

        wrapped.step = recorded_step
        return wrapped, cfg

    runners = []

    def make_runner(*positional, **keywords):
        runner, cfg = original_make_runner(*positional, **keywords)
        runners.append(runner)
        return runner, cfg

    task_registry.make_env, task_registry.make_alg_runner = make_env, make_runner
    try:
        diagnose(argparse.Namespace(output=args.output/"rollout", checkpoint=args.checkpoint,
            num_envs=args.num_envs, seed=args.seed, seconds=20., device=args.device,
            allow_recovery_change=False))
        r = recorder[0]
        assert r.count == 10000 and len(r.qpos) == 2001
        assert all(torch.equal(frozen_actor[0][k], v) for k, v in runners[0].alg.actor.state_dict().items())
        rollout = json.loads((args.output/"rollout/result.json").read_text())
        with np.load(args.output/"rollout/measurements.npz") as old:
            weight = old["weight_N"]
            initial_qpos, initial_qvel = old["initial_qpos"], old["initial_qvel"]
        trace = r.trace.cpu().numpy()
        np.savez_compressed(args.output/"physical.npz", values=trace, fields=np.asarray(FIELDS),
            weight_N=weight, initial_qpos=initial_qpos, initial_qvel=initial_qvel,
            time_s=(np.arange(10000)+1)*.002)
        np.savez_compressed(args.output/"qpos.npz", qpos=torch.stack(r.qpos).cpu().numpy(),
            time_s=np.arange(2001)*.01)
        rows = summarize(trace, weight, rollout, .002)
        passed = [x for x in rows if x["passed"]]
        result = dict(criterion=CRITERION, task="chuanliantui_stand_v2", duration_s=20.,
            num_envs=args.num_envs, seed=args.seed, physics_dt_s=.002, policy_dt_s=.01,
            parameter_updates=0, actor_and_encoder_unchanged=True, drift_is_failure_gate=False,
            passed=len(passed), ever_stood=sum(x["ever_stood"] for x in rows),
            final_balanced=sum(x["final_balanced"] for x in rows),
            ever_inverted_envs=[x["env"] for x in rows if x["ever_inverted"]],
            balanced_last5_median_speed_m_s=(float(np.median([x["last5_median_speed_m_s"]
                for x in passed])) if passed else None),
            checkpoint=rollout["checkpoint"], checkpoint_sha256=rollout["checkpoint_sha256"],
            xml_sha256=rollout["xml_sha256"], sources=rollout["sources"],
            source_sha256=digest(__file__), physical_sha256=digest(args.output/"physical.npz"),
            qpos_sha256=digest(args.output/"qpos.npz"), rows=rows,
            scope="Single-seed nominal ground-start 20s acceptance; no noise/delay randomization or hardware claim. "
                  "Final 1s balance sampled at 500Hz; rise is the unchanged task success flag. "
                  "Contact sensors retain normal simulator pre-integration timing; no extra forward calls.")
        (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        (args.output/"source.py").write_bytes(Path(__file__).read_bytes())
        print(json.dumps({k:v for k,v in result.items() if k not in ("sources", "rows")}, indent=2))
    finally:
        task_registry.make_env, task_registry.make_alg_runner = original_make_env, original_make_runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
