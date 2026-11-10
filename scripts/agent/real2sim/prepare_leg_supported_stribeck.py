"""Seed static/sliding resistance with an explicitly supported extension end; no integration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    root = Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    raw = [inverse.load('wide', root/'inverse-preload-hypothesis'), inverse.load('slow', root/'inverse-near-slow'),
           inverse.load('fast', root/'inverse-near-fast'),
           inverse.load('original', Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original'))]
    data = copy.deepcopy(raw)
    for d in data:
        # 只在种子回归排除伸长端；GPU仍从原始起点完整积分，不排除任何评分样本。
        for j in range(4): d['velocity'][d['compression'][:, j//2] < .008, j] = 0.
    inverse.ALL = data
    fits = []
    for common, damping, fixed in [(False, False, None), (False, True, None), (False, False, 150.), (True, False, None)]:
        fits.append(inverse.fit(data, common_spring=common, fit_damping=damping, fixed_reference=fixed))
    candidates = []
    for fit in fits:
        p = fit['parameters']; capacities = []
        for j in range(4):
            loads = []
            for d in raw:
                force = np.asarray(p['spring_reference_force_n'])+np.asarray(p['spring_stiffness_n_m'])*(d['compression']-.005)-np.asarray(p['spring_damping_ns_m'])*d['qvel'][:, inverse.SLIDE_INDEX]
                residual = d['target']-np.einsum('tij,tj->ti', d['spring_basis_nm_per_n'], force)
                quiet = (abs(d['velocity'][:, j]) < .005) & (d['compression'][:, j//2] >= .008)
                quiet[:10] = quiet[-10:] = False
                loads.extend(abs(residual[quiet, j]).tolist())
            capacities.append(max(float(np.percentile(loads, 90)), p['motor_dynamic_positive_nm'][j], p['motor_dynamic_negative_nm'][j]))
        for static_scale in (.75, 1., 1.25):
            for speed in (.005, .015, .04, .1):
                profile = copy.deepcopy(p)
                profile.update(friction_nm=np.maximum(np.asarray(capacities)*static_scale,
                    np.maximum(p['motor_dynamic_positive_nm'], p['motor_dynamic_negative_nm'])).tolist(),
                    damping_nms_rad=np.maximum(p['damping_nms_rad'], .05).tolist(),
                    motor_velocity_scale_rad_s=[speed]*4, armature_kg_m2=[.015795]*4, spring_force_limit_n=[600.]*2,
                    spring_extension_stop_m=[.00737, .00737745], spring_extension_margin_m=[.001]*2)
                for key in ('motor_dynamic_positive_nm', 'motor_dynamic_negative_nm'):
                    profile[key] = np.maximum(profile[key], .02).tolist()
                profile['friction_nm'] = np.maximum(profile['friction_nm'], np.maximum(profile['motor_dynamic_positive_nm'], profile['motor_dynamic_negative_nm'])).tolist()
                candidates.append(profile)
    (args.output/'parameters.json').write_text(json.dumps(candidates, indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(dict(scope='Conditional inverse seeds only; extension support is a diagnostic hypothesis',
        environment_steps=0, seed_only_minimum_compression_m=.008, full_gpu_scoring_must_include_original_start=True,
        fits=fits, candidates=len(candidates), sources=[dict(name=d['name'], source=d['source'],
        sha256=hashlib.sha256((Path(d['source'])/'inverse-loads.npz').read_bytes()).hexdigest()) for d in raw]), indent=2)+'\n')
    print(json.dumps(dict(candidates=len(candidates), fits=[f['parameters'] for f in fits]), indent=2))


if __name__ == '__main__': main()
