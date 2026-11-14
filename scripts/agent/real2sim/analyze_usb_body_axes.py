"""Compare body roll/pitch spectra in recorded data and saved GPU diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.signal import welch


def describe(angles, fs):
    result = {}
    for index, axis in enumerate(('roll', 'pitch')):
        values = angles[:, index]
        f, power = welch(values, fs=fs, nperseg=round(4*fs),
                         noverlap=round(3*fs), detrend='linear')
        band = (f >= 3) & (f < 10)
        all_band = (f >= .5) & (f < 50)
        result[axis] = dict(peak_to_peak_deg=float(np.ptp(values)),
                            band_3_10Hz_RMS_deg=float(np.sqrt(power[band].sum()*(f[1]-f[0]))),
                            dominant_Hz=float(f[all_band][np.argmax(power[all_band])]),
                            band_peak_Hz=float(f[band][np.argmax(power[band])]))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--simulation', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.experiment/'rl_valid.csv'
    with source.open() as stream:
        reader = csv.reader(stream)
        names = next(reader)
        data = np.asarray(list(reader), dtype=float)
    columns = {name: data[:, i] for i, name in enumerate(names)}
    # IMU观测100Hz发布；去掉500Hz日志中重复持有的同一帧。
    unique = np.r_[True, np.diff(columns['obs_sequence']) != 0]
    time = (columns['h7_us']-columns['h7_us'][0])/1e6
    mask = unique & (time >= 1)
    gravity = np.stack([columns[f'obs_{i}'][mask] for i in (3, 4, 5)], axis=1)
    gravity /= np.linalg.norm(gravity, axis=1, keepdims=True)
    angles = np.rad2deg(np.c_[np.arctan2(-gravity[:, 1], -gravity[:, 2]),
                              np.arcsin(gravity[:, 0].clip(-1, 1))])
    results = dict(real=dict(scope='t>=1s, mixed driving commands, unique 100Hz observations',
                            observations=len(angles), **describe(angles, 100)))
    inputs = [source]
    simulation = args.simulation or args.experiment/'high-frequency-stable-01'
    names = [row['name'] for row in json.loads((simulation/'plan.json').read_text())['profiles']] if args.simulation else ('baseline', 'leg_lag_30ms')
    for name in names:
        path = simulation/'traces'/(name+'.npz')
        inputs.append(path)
        with np.load(path) as trace:
            mask = (trace['time'] >= 20) & (trace['time'] < 30)
            results[name] = dict(scope='20-30s, zero command, saved GPU trace',
                                 **describe(trace['roll_pitch_deg'][mask], 500))
    results['limitations'] = 'Different command conditions; axis spectra characterize motion, not its cause. No new physics steps or training.'
    results['sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    (args.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    shutil.copy2(__file__, args.output/'analysis-source.py')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
