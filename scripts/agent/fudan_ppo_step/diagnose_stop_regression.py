"""Compare zero-command motor traces and isolate braking actions without training."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from ct_mjlab.fudan_ppo_walk import policy as policy_module
from . import diagnose_reverse
from .launch_training import ROOT, sha, write
from .prepare_restart import SOURCE, SOURCE_SHA
from .review_restart import BASELINE, native_trial

TRAINING = ROOT/'artifacts/fudan_ppo_step/20261119-restart-v11-5200-to6200-01'
AFTER = TRAINING/'regression/gpu-stair-6200'
WHEELS = [2, 5]
LEGS = [0, 1, 3, 4]
PARTS = {'wheel': WHEELS, 'leg': LEGS, 'full': list(range(6))}


def new_checkpoint():
    audit = json.loads((TRAINING/'verification/training-audit.json').read_text())
    path = Path(audit['checkpoint'])
    assert audit['passed'] and audit['iterations'] == 6200
    assert sha(path) == audit['checkpoint_sha256'] and sha(SOURCE) == SOURCE_SHA
    return path


def read_trace(folder, case, protocol):
    path = folder/'traces'/f'{case}.npz'
    assert sha(path) == protocol['trace_sha256'][case]
    with np.load(path) as trace:
        return dict(trace)


def parking_start(trace):
    return np.flatnonzero(trace['valid'] & (trace['command'][:, :2] == 0).all(axis=1))[0]


def fixed_cases(protocol):
    return [case['id'] for case in protocol['cases'] if case['height'] == .2 and case['speed'] == 2.]


def offline(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, folder in ((5200, BASELINE), (6200, AFTER)):
        p = json.loads((folder/'protocol.json').read_text())
        model = mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
        rates = [int(model.joint(f'robot/{wheel}').dofadr[0]) for wheel in ('lfwheel', 'rfwheel')]
        for case in fixed_cases(p):
            trace = read_trace(folder, case, p)
            stop = parking_start(trace)
            # Compare the first two seconds after the same zero-command event.
            end = min(stop+200, int(trace['valid'].sum()))
            windows = {'braking': slice(stop, stop+50), 'holding': slice(stop+50, end)}
            for name, window in windows.items():
                torque = trace['torque_sub'][window]
                wheel_rate = trace['qvel'][window][:, rates]
                speed = np.linalg.norm(trace['linear_world'][window, :2], axis=1)
                window_torque = torque[:, :, WHEELS]
                angle = trace['gravity'][window]
                rows.append(dict(iterations=label, case=case, window=name, stop_command_s=stop*.01,
                    speed_rms_m_s=float(np.sqrt(np.mean(speed**2))), speed_peak_m_s=float(speed.max()),
                    wheel_rate_rms_rad_s=float(np.sqrt(np.mean(wheel_rate**2))),
                    wheel_torque_rms_Nm=float(np.sqrt(np.mean(window_torque**2))),
                    wheel_torque_mean_Nm=window_torque.mean((0, 1)).tolist(),
                    leg_torque_rms_Nm=float(np.sqrt(np.mean(torque[:, :, LEGS]**2))),
                    wheel_drive_power_at_policy_end_W=float(np.mean(np.sum(window_torque[:, -1]*wheel_rate, axis=1))),
                    supported_fraction=float(trace['normal_support'][window].mean()),
                    tilt_peak_deg=float(np.degrees(np.arccos((-angle[:, 2]).clip(-1, 1))).max())))
    write(output/'torque-comparison.json', dict(rows=rows, optimizer_updates=0,
        windows_s={'braking': [0., .5], 'holding': [.5, 2.]},
        power_note='Torque and instantaneous wheel speed at the last substep; diagnostic sample power.'))
    for window in ('braking', 'holding'):
        for iteration in (5200, 6200):
            batch = [r for r in rows if r['window'] == window and r['iterations'] == iteration]
            fields = ('wheel_rate_rms_rad_s', 'wheel_torque_rms_Nm', 'leg_torque_rms_Nm',
                      'speed_peak_m_s', 'wheel_drive_power_at_policy_end_W', 'supported_fraction', 'tilt_peak_deg')
            print(window, iteration, {key:round(float(np.median([r[key] for r in batch])),4) for key in fields})


class StopBlend:
    """Use selected 5200 action components after a prior forward command ends."""

    def __init__(self, actor, parking, components):
        self.actor = actor
        self.parking = parking
        self.components = components
        self.started = None

    def state_dict(self):
        return self.actor.state_dict()

    def act_inference(self, obs, history):
        action, latent = self.actor.act_inference(obs, history)
        forward = obs[:, 6] > 0.
        if self.started is None:
            self.started = torch.zeros_like(forward)
        self.started |= forward
        use_parking = self.started & obs[:, 6:8].eq(0.).all(dim=1)
        if use_parking.any():
            backup, _ = self.parking.act_inference(obs, history)
            for index in self.components:
                action[:, index] = torch.where(use_parking, backup[:, index], action[:, index])
        return action, latent


def substitute(checkpoint, components):
    real_load = policy_module.load_policy
    def load(path, device='cpu'):
        actor, saved = real_load(path, device=device)
        if Path(path).resolve() == checkpoint.resolve() and components:
            parking, _ = real_load(SOURCE, device=device)
            actor = StopBlend(actor, parking, components)
        return actor, saved
    return load


def gpu(output, variant):
    checkpoint = new_checkpoint()
    output.mkdir(parents=True, exist_ok=False)
    real_config = diagnose_reverse.FudanStepCfg
    def evaluation_config():
        cfg = real_config()
        cfg.step.patch_size = (10., 4.)
        cfg.step.flat_forward_limit_m = 3.5
        return cfg
    request = SimpleNamespace(output=output, checkpoint=checkpoint, command_height=.35,
        label=f'6200+5200-stop-{variant}', seconds=20., deadline=time.time()+600, heights=[.2],
        speeds=[2.], angles=[0., -10., 10.], directions=['forward'], friction=.8, high_speed=True,
        frictions=[.4, .8, 1.2])
    components = PARTS.get(variant, [])
    write(output/'intervention.json', dict(restoring_components=components, start_sha256=sha(checkpoint),
        stop_sha256=SOURCE_SHA if components else None, switching_condition='Prior forward and zero vx,yaw',
        script_sha256=sha(__file__)))
    with patch.object(policy_module, 'load_policy', substitute(checkpoint, components)), \
         patch.object(diagnose_reverse, 'FudanStepCfg', evaluation_config):
        diagnose_reverse.run(request)
    audit = output/'audit'
    audit.mkdir()
    diagnose_reverse.audit(SimpleNamespace(recording=output, output=audit))
    protocol = json.loads((output/'protocol.json').read_text())
    frozen = json.loads((AFTER/'protocol.json').read_text())
    before_error, timing = {}, {}
    rows = [r for r in json.loads((output/'metrics.json').read_text()) if r['height'] == .2]
    for row in rows:
        trace = read_trace(output, row['id'], protocol)
        initial = read_trace(AFTER, row['id'], frozen)
        assert float(abs(trace['initial']-initial['initial']).max())<2e-6
        stop = parking_start(trace)
        old_stop = parking_start(initial)
        end = min(stop, old_stop)
        before_error[row['id']] = float(abs(trace['qpos'][:end]-initial['qpos'][:end]).max())
        timing[row['id']] = [old_stop*.01, stop*.01]
    result = dict(variant=variant, landed=sum(r['landed'] for r in rows),
        passed=sum(r['task_score']['passed'] for r in rows), total=len(rows),
        per_case={r['id']:dict(passed=r['task_score']['passed'], parking_s=r['task_score']['stable_stop_s'],
                              termination=r['stop_reason']) for r in rows},
        approach_position_max_error_relative_to_48_world_recording=before_error, parking_switch_time_s=timing,
        optimizer_updates=0, full_task_accepted=False)
    write(output/'result.json',result)
    print(json.dumps(result))


def native(output, variant):
    from . import review_restart
    checkpoint = new_checkpoint()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    with patch.object(review_restart, 'load_policy', substitute(checkpoint, PARTS[variant])):
        for seed in (1, 2, 3):
            for wait in (0., 1.):
                rows.append(native_trial(checkpoint,'flat',wait,seed,output/f'flat-{wait:g}-{seed}.npz',
                    stop_timeout_s=8., stop_check_from_s=.2))
    write(output/'summary.json',dict(cases=rows, variant=variant, checkpoint_sha256=sha(checkpoint),
        diagnostic=True, optimizer_updates=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('offline','gpu','native'))
    p.add_argument('--variant', choices=('control', *PARTS), default='wheel')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'offline':
        offline(args.output)
    else:
        {'gpu':gpu,'native':native}[args.mode](args.output,args.variant)
