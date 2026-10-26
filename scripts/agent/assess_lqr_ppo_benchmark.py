#!/usr/bin/env python3
"""Independent contact/trajectory metrics for cached-pose LQR and PPO runs."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import torch


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(directory, cached):
    r = json.loads((directory/'result.json').read_text())
    assert r['completed'] and r['optimizer_updates'] == 0
    assert r['trace_sha256'] == digest(directory/'trace.npz')
    with np.load(directory/'trace.npz') as archive: d = {k: archive[k] for k in archive.files}
    count = r['num_envs']; dt = r['physical_dt']; f = d['physical']
    if r['kind'] != 'native':
        assert 'terminated' in d and 'time_out' in d, 'Need physical failure separate from normal time limit'
    np.testing.assert_array_equal(d['initial_qpos'][:, 2:], cached['qpos'].numpy()[:count, 2:])
    np.testing.assert_array_equal(d['initial_qvel'], cached['qvel'].numpy()[:count])
    assert f.shape == (round(20/dt), count, 12) and np.isfinite(f).all()
    w = r['weight_n']; tilt = np.maximum(np.abs(f[:, :, 3]), np.abs(f[:, :, 4]))
    balanced = ((f[:, :, :2] >= max(.1*w, 5.)).all(-1) & (f[:, :, :2].sum(-1) >= .8*w)
        & (f[:, :, :2].sum(-1) <= 1.2*w) & (f[:, :, 2] <= 1.)
        & (tilt <= math.radians(35)) & (f[:, :, 5] <= .5) & (f[:, :, 7] > .5))
    strict = balanced & (tilt <= math.radians(15)) & (np.abs(f[:, :, 6]-.33648) <= .03)
    run = np.zeros(count, dtype=int); peak = run.copy(); hold = run.copy()
    for j in range(len(f)):
        run = np.where(strict[j], run+1, 0); peak = np.maximum(peak, run)
        hold = np.where(balanced[j], hold+1, 0)
    for row in r['rows']:
        i = row['env']; n = int(d['alive'][:, i].sum()); late = slice(1500, n)
        if r['kind'] != 'native':
            assert bool(d['terminated'][:, i].any()) == row['terminated']
        full = n == 2000
        assert bool(peak[i] >= round(.5/dt)) == row['strict_rise']
        assert bool(full and not row['terminated'] and hold[i] >= round(1./dt)) == row['final_balance_v31']
        if n > 1500:
            metrics = dict(last5s_leg_velocity_rms_rad_s=float(np.sqrt(np.square(d['motor_dq'][late, i][:, [0, 1, 3, 4]]).mean())),
                last5s_mean_speed_m_s=float(np.linalg.norm(d['features'][late, i, 4:6], axis=-1).mean()),
                last5s_xy_excursion_m=float(np.linalg.norm(d['qpos'][late, i, :2]-d['qpos'][1500, i, :2], axis=-1).max()),
                last5s_mean_abs_yaw_rate_rad_s=float(np.abs(d['features'][late, i, 9]).mean()))
            for name, value in metrics.items(): assert abs(value-row[name]) < 1e-9
        if full:
            physical_leg = float(np.sqrt(np.square(f[round(15/dt):, i, 8:12]).mean()))
            assert abs(physical_leg-row['last5s_physical_leg_velocity_rms_rad_s']) < 1e-6
        values = (row['last5s_leg_velocity_rms_rad_s'], row['last5s_mean_speed_m_s'],
                  row['last5s_xy_excursion_m'], row['last5s_mean_abs_yaw_rate_rad_s'])
        qualified = bool(full and row['strict_rise'] and row['final_balance_v31'] and row['finite']
                         and all(v is not None and v <= limit for v, limit in zip(values, (.02, .01, .02, .02))))
        assert qualified == row['qualified']
    stats = {key: float(np.mean([row[key] for row in r['rows'] if row[key] is not None]))
        for key in ('last5s_leg_velocity_rms_rad_s', 'last5s_mean_speed_m_s', 'last5s_xy_excursion_m',
                    'last5s_physical_leg_velocity_rms_rad_s') if any(row[key] is not None for row in r['rows'])}
    return dict(kind=r['kind'], strict_rise=r['strict_rise'], final_balance=r['final_balance'],
                qualified=r['qualified'], count=count, means=stats, trace_sha256=r['trace_sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); cached = torch.load(args.source/'nominal_mean/initial-state.pt', map_location='cpu', weights_only=False)
    result = dict(passed=True, optimizer_updates=0, groups={str(run): assess(run, cached) for run in args.run},
        scope='Recomputed all physical-step v3.1 support and strict-rise durations,100Hz motion and full-rate leg RMS; identical cached poses, native1ms/0s hold versus student2ms/0.2s hold explicitly differ.')
    args.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))
