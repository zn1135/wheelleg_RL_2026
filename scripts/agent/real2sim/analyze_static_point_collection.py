"""Turn frozen collected static points into same-pose load reports; no physics integration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from scripts.agent.real2sim import check_leg_static_load as static
from scripts.agent.real2sim.analyze_tilt_length import geometry
from ct_mjlab.envs.chuanliantui import gas_spring


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--collections', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    grouped, sources = {}, []
    identity = None
    for directory in args.collections:
        plan = json.loads((directory / 'plan.json').read_text())
        pair = (plan['model']['sha256'], plan['config']['sha256'])
        if identity is None:
            identity = pair
            model = static.ROOT / plan['model']['project_path']
            config = static.ROOT / plan['config']['project_path']
        assert identity == pair
        content = (directory / 'points.jsonl').read_bytes()
        content = content[:content.rfind(b'\n') + 1]
        sources.append(dict(collection=str(directory.resolve()), ledger_prefix_bytes=len(content),
                            ledger_prefix_sha256=hashlib.sha256(content).hexdigest(), sampling_mode=plan.get('sampling_mode')))
        for line in content.splitlines():
            point = json.loads(line)
            data = directory / point['data_file']
            assert digest(data) == point['data_sha256']
            group = grouped.setdefault(point['data_sha256'], dict(path=data, points=[]))
            group['points'].append(point)
    assert grouped, 'No recorded static points'
    assert digest(model) == identity[0] and digest(config) == identity[1]
    pose = static.mirror.MirrorPose(model, config)
    pose.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    import re
    block = config.read_text().split('[MACHINE_ID_BIG_WHEELLEG] = {', 1)[1].split('[MACHINE_ID_SMALL_WHEELLEG]', 1)[0]
    lu, lg = [float(re.search(r'\.' + n + r'\s*=\s*([\d.]+)f', block)[1]) for n in ('leg_lu', 'leg_lg')]
    reports, observations = [], []
    for index, (data_hash, group) in enumerate(grouped.items(), 1):
        f = np.load(group['path'])['frames']
        q = np.median(f[:, 10:14], axis=0)
        feedback = np.mean(f[:, 146:150], axis=0)
        gravity, spring, passive, checks = static.pose_load(pose, f[len(f) // 2], q)
        sign = pose.sign[:4]
        gravity, spring, passive = gravity * sign, spring * sign[:, None], passive * sign
        sq = [pose.model.joint(s + '_spring2_joint').qposadr[0] for s in ('left', 'right')]
        compression = np.maximum(.010 - pose.data.qpos[sq], 0.)
        nominal = gas_spring.gas_spring_force(pose.data.qpos[sq], np.zeros(2))
        legs = []
        for side, label in enumerate(('left', 'right')):
            length, angle = geometry(q[None, 2 * side:2 * side + 2], lu, lg)
            b = spring[2 * side:2 * side + 2, side]
            net = (feedback - gravity - passive)[2 * side:2 * side + 2]
            force = float(b @ net / (b @ b))
            legs.append(dict(side=label, length_m=float(length[0]), leg_angle_rad=float(angle[0]),
                             spring_compression_m=float(compression[side]), spring_required_nm_per_n=b.tolist(),
                             model_spring_force_n=float(nominal[side]), unconstrained_equivalent_spring_force_n=force,
                             measured_minus_best_model_nm=(net - b * max(0., force)).tolist()))
        fit_sides = sorted({p['side'] for p in group['points']})
        report = dict(scope='Frozen static point collection; no integration; equivalent forces are conditional',
                      model_sha256=identity[0], config_sha256=identity[1], source_frames_sha256=data_hash,
                      source_points=group['points'], source_frame_file=str(group['path'].resolve()), fit_sides=fit_sides,
                      samples=len(f), board_time_s=[static.mirror.read_u32(f[0, 8:10]) / 1000,
                                                  static.mirror.read_u32(f[-1, 8:10]) / 1000],
                      motor_positions_rad=q.tolist(), feedback_mean_nm=feedback.tolist(),
                      request_mean_nm=np.mean(f[:, 132:136], axis=0).tolist(),
                      gravity_required_nm=gravity.tolist(), passive_required_nm=passive.tolist(),
                      model_total_required_nm=(gravity + passive + spring @ nominal).tolist(),
                      legs=legs, verification=checks, physics_steps=0)
        path = args.output / f'pose-{index:03d}.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        reports.append(str(path.resolve()))
        for point in group['points']:
            side = ('left', 'right').index(point['side'])
            row = dict(report=str(path.resolve()), side=point['side'], direction=point['direction'],
                       actual_length_mm=point['actual_length_mm'], **{k: legs[side][k] for k in
                       ('spring_compression_m', 'unconstrained_equivalent_spring_force_n', 'measured_minus_best_model_nm')})
            observations.append(row)
        print(f'Analyzed static pose {index}/{len(grouped)}', flush=True)
    summary = dict(sources=sources, unique_poses=len(grouped), side_observations=len(observations),
                   reports=reports, observations=observations, model_sha256=identity[0], config_sha256=identity[1],
                   physics_steps=0, limitation='No unique spring/friction identification or dynamic acceptance; only recorded sides are included in fits')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for source in (Path(__file__), Path(static.__file__), Path(gas_spring.__file__), model, config):
        destination = args.output / 'source' / source.resolve().relative_to(static.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


if __name__ == '__main__':
    main()
