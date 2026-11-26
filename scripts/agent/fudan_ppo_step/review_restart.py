"""Assess T handoff, stop/restart, and paired 200 mm GPU stair parking."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from ct_mjlab.fudan_ppo_walk.policy import load_policy
from sim2sim.rmuc2027_arena import ArenaPlayback, REGIONAL_CHECKPOINT
from sim2sim.step_playground import ManualStepPolicies, build_model, layout
from .launch_training import ROOT, sha, write
from .prepare_restart import SOURCE, SOURCE_SHA
from .review_fixed_speed import stairs

BASELINE = ROOT/'artifacts/fudan_ppo_step/20261118-fixed-speed2-remote-handoff-01/regression/stair-5200'


def native_trial(checkpoint, surface, wait, seed, output, *, stop_timeout_s=3., stop_check_from_s=3.):
    actor, saved = load_policy(checkpoint)
    switch = ManualStepPolicies(REGIONAL_CHECKPOINT, checkpoint, actor, saved)
    model = build_model(saved)
    env = ArenaPlayback(model, saved, seed=seed)
    spot = next(s for s in layout(saved) if s['height'] == (.2 if surface == 'stair' else 0.))
    env.command[:] = (0., 0., .26)
    env.reset((*spot['spawn'], 0.), yaw=0.)
    wheels = [model.body(name).id for name in ('lfwheel', 'rfwheel')]
    rows = []
    rotation = np.empty(9)

    def advance():
        obs = env.observation()
        env.history[-19:-16] = obs[6:9]
        action, _ = switch.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(env.history[None]))
        env.step(action[0].numpy())

    def record(duration, target):
        env.command[:2] = (target, 0.)
        result = []
        for _ in range(round(duration/env.policy_dt)):
            advance()
            mujoco.mju_quat2Mat(rotation, env.data.qpos[3:7])
            vx = rotation.reshape(3, 3)[:, 0] @ env.data.qvel[:3]
            result.append((env.data.time, target, vx, np.linalg.norm(env.data.qvel[:2]),
                           env.data.qpos[2], *env.data.xpos[wheels, 0]))
        values = np.asarray(result)
        rows.append(values)
        return values

    def speed_timing(values):
        eligible = np.ones(len(values), bool)
        if surface == 'stair':
            eligible &= values[:, -2:].min(1) < spot['edge'][0]
        at_speed = (values[:, 2] >= 1.9)&eligible
        steady = np.convolve(at_speed.astype(int), np.ones(10, int), mode='valid') == 10
        def first(mask):
            hit = np.flatnonzero(mask)
            return float((hit[0]+1)*env.policy_dt) if hit.size else None
        return dict(first_1_9_s=first(at_speed), sustained_1_9_onset_s=first(steady),
                    peak_vx_m_s=float(values[eligible, 2].max()) if eligible.any() else None)

    with torch.inference_mode():
        walk_hold = record(2., 0.)
        while not (walk_hold[-20:, 3] <= .05).all() and env.data.time < 6.:
            walk_hold = record(.20, 0.)
        settled_at_t = bool((walk_hold[-20:, 3] <= .05).all())
        switch.select('step', env)  # Same method as the T key.
        if wait:
            record(wait, 0.)
        first = record(2.5, 2.)
        stopped = record(stop_check_from_s, 0.)
        stop_samples = len(stopped)
        last_stop_speeds = stopped[-20:, 3].copy()
        if surface == 'flat':
            assert len(last_stop_speeds) == 20
            while stop_samples < round(stop_timeout_s/env.policy_dt) and not (last_stop_speeds <= .05).all():
                row = record(env.policy_dt, 0.)
                last_stop_speeds = np.r_[last_stop_speeds[1:], row[0, 3]]
                stop_samples += 1
        rearm = surface == 'flat' and bool((last_stop_speeds <= .05).all())
        restart = speed_timing(record(3., 2.)) if rearm else None
    np.savez_compressed(output, data=np.concatenate(rows), columns=['time_s', 'target_vx_m_s', 'actual_vx_m_s',
        'horizontal_speed_m_s', 'body_height_m', 'left_wheel_x_m', 'right_wheel_x_m'])
    return dict(iterations=saved['iterations'], surface=surface, t_to_up_s=wait, seed=seed,
        walk_settled=settled_at_t,
        first_start=speed_timing(first), stopped_before_restart=rearm, restart=restart,
        stop_observation_s=stop_samples*env.policy_dt,
        stop_window_peak_m_s=float(last_stop_speeds.max()),
        action_delay_ms=int(env.delays.action_steps[0]), sensor_delay_ms=int(env.delays.sensor_steps[0]),
        transitions=switch.transitions)


def run(audit_path, output):
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['iterations'] == 6200
    checkpoint = Path(audit['checkpoint'])
    assert sha(checkpoint) == audit['checkpoint_sha256'] and sha(SOURCE) == SOURCE_SHA
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    samples = []
    for tick, model in ((5200, SOURCE), (6200, checkpoint)):
        for seed in (1, 2, 3):
            for surface in ('flat', 'stair'):
                for wait in (0., 1.):
                    samples.append(native_trial(model, surface, wait, seed,
                        output/f'native-{tick}-{surface}-{wait:g}-{seed}.npz'))
    write(output/'native-start.json', dict(cases=samples, optimizer_updates=0, formal_acceptance=False))
    after = output/'gpu-stair-6200'
    stairs(checkpoint, after, time.time()+1200)
    def rows(folder):
        return {row['id']: row for row in json.loads((folder/'metrics.json').read_text()) if row['height'] == .2}
    old, new = rows(BASELINE), rows(after)
    assert old.keys() == new.keys() and len(old) == 36
    for folder in (BASELINE, after):
        assert json.loads((folder/'audit/audit.json').read_text())['passed']
    for key in ('cases', 'seconds', 'command_height_m', 'action_delay_steps', 'sensor_delay_steps', 'friction'):
        assert json.loads((BASELINE/'protocol.json').read_text())[key] == json.loads((after/'protocol.json').read_text())[key], key
    max_initial_difference = 0.
    for case in old:
        with np.load(BASELINE/'traces'/f'{case}.npz') as a, np.load(after/'traces'/f'{case}.npz') as b:
            max_initial_difference = max(max_initial_difference, float(abs(a['initial']-b['initial']).max()))
    assert max_initial_difference < 2e-6
    old_pass = {key for key, row in old.items() if row['task_score']['passed']}
    new_pass = {key for key, row in new.items() if row['task_score']['passed']}
    target = {key for key, row in old.items() if row['speed'] == 2.}
    assert len(target) == 9 and target <= old_pass
    write(output/'review.json', dict(checkpoint_sha256=audit['checkpoint_sha256'], iterations=6200,
        native_cases=samples, fixed_2mps_stair_passed=len(new_pass&target),
        fixed_2mps_stair_regressed=sorted(target-new_pass), other_stair_regressed=sorted((old_pass-new_pass)-target),
        stair_2mps_regression_passed=target<=new_pass, max_initial_difference=max_initial_difference,
        full_task_accepted=False, optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.audit, args.output)
