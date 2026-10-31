#!/usr/bin/env python3
"""Compare a simulation-only leg-target handover with the same trained policy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand_v2.action import StandV2PdAction
from scripts.agent.stand_v2.diagnose_chuanliantui_stand_v2 import diagnose


def main(args):
    original = json.loads(args.report.read_text())
    assert args.handover_seconds >= 0.
    output = args.output
    instances, runners, actor_before = [], [], {}

    class DiagnosticHandover(StandV2PdAction):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.start_angles = torch.zeros(self.num_envs, 4, device=self.device)
            self.handover_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.handover_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.first_second_torques, self.first_second_qvel = [], []
            self.max_target_roundtrip_error = torch.zeros((), device=self.device)
            instances.append(self)

        def reset(self, env_ids=None):
            super().reset(env_ids)
            if hasattr(self, "handover_started"):
                ids = slice(None) if env_ids is None else env_ids
                self.handover_started[ids] = False
                self.handover_ticks[ids] = 0

        def apply_actions(self):
            if not actor_before and runners:
                actor_before.update({k: v.clone() for k, v in runners[0].alg.actor.state_dict().items()})
            original_actions = None
            if args.handover_seconds:
                hold = self._ground_hold()
                ground = self._env.command_manager.get_term("standup").ground_start
                begin = ground & ~hold & ~self.handover_started
                legs = self._leg_action_ids
                current = self._env.sim.data.qpos[:, self._motor_q_adr][:, legs]
                self.start_angles.copy_(torch.where(begin[:, None], current, self.start_angles))
                self.handover_started |= begin
                fraction = (self.handover_ticks * self._env.physics_dt / args.handover_seconds).clamp(0., 1.)
                blend = fraction.square() * (3. - 2. * fraction)
                active = self.handover_started & ~hold & (fraction < 1.)
                original_actions = self._control_actions.clone()
                reference = self._reference_pos[:, legs]
                policy_target = (self._nominal_pos[:, legs]
                    + self.cfg.leg_position_scale * original_actions[:, legs]).clamp(reference-3.14, reference+3.14)
                # Only leg targets transition; the original wheel law and limits remain in force.
                target = self.start_angles + blend[:, None] * (policy_target-self.start_angles)
                equivalent_action = (target-self._nominal_pos[:, legs]) / self.cfg.leg_position_scale
                self._control_actions[:, legs] = torch.where(active[:, None], equivalent_action,
                    original_actions[:, legs])
                actual = (self._nominal_pos[:, legs]
                    + self.cfg.leg_position_scale * self._control_actions[:, legs]).clamp(reference-3.14, reference+3.14)
                expected = torch.where(active[:, None], target.clamp(reference-3.14, reference+3.14), policy_target)
                self.max_target_roundtrip_error = torch.maximum(self.max_target_roundtrip_error,
                    (actual-expected).abs().max())
            super().apply_actions()
            if original_actions is not None:
                self._control_actions.copy_(original_actions)
                self.handover_ticks += (self.handover_started & ~hold).long()
            # First-second substep records measure the release impulse at the real 500 Hz.
            if len(self.first_second_torques) < round(1. / self._env.physics_dt):
                self.first_second_torques.append(self.torques.clone())
                self.first_second_qvel.append(self._env.sim.data.qvel.clone())

    get_cfgs, make_runner = task_registry.get_cfgs, task_registry.make_alg_runner

    def diagnostic_cfgs(name, *, play=False):
        cfg, agent = get_cfgs(name, play=play)
        if name == "chuanliantui_stand_v2" and play:
            cfg.actions["pd"].class_type = DiagnosticHandover
        return cfg, agent

    def capture_runner(*positional, **keywords):
        runner, agent = make_runner(*positional, **keywords)
        runners.append(runner)
        return runner, agent

    task_registry.get_cfgs, task_registry.make_alg_runner = diagnostic_cfgs, capture_runner
    try:
        run_args = argparse.Namespace(output=output, num_envs=original["num_envs"],
            seed=original["seed"], seconds=original["duration_s"], device="cuda:0",
            checkpoint=Path(original["checkpoint"]), allow_recovery_change=False)
        diagnose(run_args)
        assert len(instances) == len(runners) == 1
        action = instances[0]
        assert actor_before
        assert all(torch.equal(actor_before[k], v) for k, v in runners[0].alg.actor.state_dict().items())
        assert float(action.max_target_roundtrip_error) < 2e-6
        torques = torch.stack(action.first_second_torques).cpu().numpy()
        qvel = torch.stack(action.first_second_qvel).cpu().numpy()
        np.savez_compressed(output/"first-second-control.npz", torque_Nm=torques,
            qvel=qvel, motor_v_adr=action._motor_v_adr.cpu().numpy())
        with np.load(args.report.parent/"measurements.npz") as old, np.load(output/"measurements.npz") as new:
            for name in ("initial_qpos", "initial_qvel", "weight_N", "fields"):
                assert np.array_equal(old[name], new[name]), name
            values, active = new["values"], new["active"]
        tilt = np.arccos(np.clip(np.cos(values[:, :, 3])*np.cos(values[:, :, 4]), -1., 1.))
        valid = active.copy()
        valid[:20] = False
        inverted = (tilt > np.pi/2) & valid
        old_success = {row["env"] for row in original["episodes"] if row["final_balance_hold_s"] >= 1.-1e-5}
        report_path = output/"result.json"
        result = json.loads(report_path.read_text())
        new_success = {row["env"] for row in result["episodes"] if row["final_balance_hold_s"] >= 1.-1e-5}
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        result["diagnostic_handover"] = dict(duration_s=args.handover_seconds,
            definition="q_target=(1-s)*q_at_release+s*clipped_policy_target; s=3*u^2-2*u^3; u=elapsed/duration",
            only_during_first_ground_start_release=True, original_position_limits_preserved=True,
            wheel_control_law_unchanged=True, actor_and_encoder_unchanged=True,
            max_target_roundtrip_error_rad=float(action.max_target_roundtrip_error),
            original_report_sha256=sha(args.report), exact_initial_state_match=True,
            ever_inverted_envs=int(inverted.any(0).sum()), inverted_env_ids=np.where(inverted.any(0))[0].tolist(),
            inverted_sample_fraction=float(inverted.sum()/valid.sum()),
            first_second_peak_leg_torque_Nm=np.abs(torques[:, :, (0, 1, 3, 4)]).max((0, 2)).tolist(),
            first_second_leg_torque_limit_envs=int((np.abs(torques[:, :, (0, 1, 3, 4)]) >= 39.99).any((0, 2)).sum()),
            first_active_substep_leg_torque_Nm=torques[100][:, (0, 1, 3, 4)].tolist(),
            lost_original_balance_envs=sorted(old_success-new_success), added_original_balance_envs=sorted(new_success-old_success),
            source_sha256=sha(__file__), parameter_updates=0,
            scope="Simulation-only control comparison; no production controller or checkpoint change; not formal acceptance")
        report_path.write_text(json.dumps(result, indent=2)+"\n")
        (output/"source.py").write_bytes(Path(__file__).read_bytes())
        print(json.dumps({"handover_seconds":args.handover_seconds, "final_balance_1s":result["final_balance_1s"],
            "ever_inverted_envs":int(inverted.any(0).sum()), "lost_original_balance_envs":sorted(old_success-new_success),
            "survived_to_end":result["survived_to_end"]}), flush=True)
    finally:
        task_registry.get_cfgs, task_registry.make_alg_runner = get_cfgs, make_runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handover-seconds", type=float, required=True)
    main(parser.parse_args())
