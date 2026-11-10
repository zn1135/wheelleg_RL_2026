"""Replay recorded torque with isolated loop/limit solver changes on MuJoCo Warp GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(model, qpos):
    data = mujoco.MjData(model)
    pairs = [pair for leg in replay.mirror.PIN_PAIRS for pair in leg]
    gaps = []; areas = []
    for q in qpos:
        data.qpos[:] = q; mujoco.mj_kinematics(model, data)
        gaps.append([np.linalg.norm(data.site(a).xpos-data.site(b).xpos) for a, b in pairs])
        row = []
        for prefix, side in [('lf', 'left'), ('rf', 'right')]:
            # 三个销点构成的有向面积；变号表示机构跨过共线姿态。
            a = data.body(prefix+'01').xpos
            b = data.body(prefix+'02').xpos
            c = data.site(side+'_front_pin1').xpos
            normal = data.body('base_link').xmat.reshape(3, 3)[:, 1]
            row.append(np.dot(np.cross(b-a, c-a), normal))
        areas.append(row)
    limited = np.flatnonzero(model.jnt_limited)
    addresses = model.jnt_qposadr[limited]
    excess = np.maximum(model.jnt_range[limited, 0]-qpos[:, addresses],
                        qpos[:, addresses]-model.jnt_range[limited, 1]).clip(0.)
    return np.asarray(gaps), np.asarray(areas), excess, limited


def metrics(model, times, qpos, measured):
    gaps, area, excess, limited = geometry(model, qpos)
    first = lambda mask: float(times[np.flatnonzero(mask)[0]]) if np.any(mask) else None
    motors = [model.joint(n).qposadr[0] for n in replay.mirror.MOTORS[:4]]
    error = np.degrees(qpos[:, motors]-measured[:, motors])
    result = dict(max_pin_gap_mm=float(gaps.max()*1000),
                  first_pin_gap_over_0_1mm_s=first(gaps.max(1)>.0001),
                  max_limit_excess_deg=float(np.degrees(excess.max())),
                  limit_excess={model.joint(j).name: dict(max_deg=float(np.degrees(excess[:, i].max())),
                      first_over_0_1deg_s=first(excess[:, i]>np.radians(.1)))
                      for i, j in enumerate(limited) if excess[:, i].max()>1e-5},
                  first_triangle_area_sign_change_s=[first(area[:, i]*area[0, i]<-1e-8) for i in (0, 1)],
                  joint_rmse_deg=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                  joint_peak_error_deg=np.max(abs(error), axis=0).tolist(),
                  joint_terminal_error_deg=abs(error[-1]).tolist())
    return result, gaps, area, excess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('artifacts/real2sim/h7-real-vs-mujoco-view-20261104-01'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', nargs='+', default=['original', 'wide'])
    parser.add_argument('--variants', nargs='+', default=['limits', 'limits_fine'])
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')
    spring = Path(replay.training_spring.__file__); protected = {str(p): sha(p) for p in (source, spring)}
    from wheel_response import prepare_wheel_response
    import mujoco_warp as mw
    import warp as wp
    wp.init(); report = dict(protected_hashes=protected, sessions={}, gpu_environment_steps=0,
                            hardware_access=False, measured_position_feedback=False,
                            spring_or_friction_changed=False, source_model_modified=False)
    for session in args.sessions:
        directory = args.output/session; directory.mkdir()
        baseline = args.baseline/session
        with np.load(baseline/'comparison.npz') as data: old = {k: data[k].copy() for k in data.files}
        reference = json.loads((baseline/'report.json').read_text())
        assert reference['source_model_sha256'] == protected[str(source)]
        with np.load(Path(reference['reference'])/'replay.npz') as data:
            initial_q = data['qpos'][0, 0].copy(); initial_v = data['qvel'][0, 0].copy()
        controls = old['submitted_model_torque_nm']; rows = {}; new_q = []; new_v = []; new_gaps = []
        source_model = mujoco.MjModel.from_xml_path(str(baseline/'source-model.xml'))
        rows['measured'] = metrics(source_model, old['time_s'], old['qpos'][:, 0], old['qpos'][:, 0])[0]
        for i, name in enumerate(reference['models']):
            rows[name] = metrics(source_model, old['time_s'], old['qpos'][:, i+1], old['qpos'][:, 0])[0]
        for variant in args.variants:
            tree = ET.parse(baseline/'source-model.xml').getroot()
            substeps = 5 if variant == 'limits_fine' else 1
            dt = .001/substeps
            for eq in tree.find('equality'):
                eq.set('solref', f'{2*dt:g} 1'); eq.set('solimp', '.999 .9999 .001 .5 2')
            for joint in tree.findall('.//joint'):
                if joint.get('range') is not None and joint.get('limited') != 'false':
                    joint.set('solreflimit', f'{2*dt:g} 1')
                    joint.set('solimplimit', '.999 .9999 .001 .5 2')
            tree.find('option').set('timestep', str(dt))
            xml = ET.tostring(tree, encoding='unicode'); (directory/(variant+'-model.xml')).write_text(xml)
            model = mujoco.MjModel.from_xml_string(xml); prepare_wheel_response(model)
            model.geom_contype[:] = model.geom_conaffinity[:] = 0
            assert np.array_equal(model.jnt_range, source_model.jnt_range)
            assert np.array_equal(model.jnt_limited, source_model.jnt_limited)
            data = mujoco.MjData(model); data.qpos[:] = initial_q; data.qvel[:] = initial_v
            motors = [model.actuator(n+'_motor').id for n in replay.mirror.MOTORS]
            springs = [model.actuator(s+'_gas_spring_motor').id for s in ('left', 'right')]
            sq = [model.joint(s+'_spring2_joint').qposadr[0] for s in ('left', 'right')]
            sv = [model.joint(s+'_spring2_joint').dofadr[0] for s in ('left', 'right')]
            steps = len(controls); q = np.empty((steps+1, 2, model.nq), dtype=np.float32)
            v = np.empty_like(q); ctrl = np.zeros((2, model.nu), dtype=np.float32)
            with wp.ScopedDevice('cuda:0'):
                gm = mw.put_model(model); gd = mw.put_data(model, data, nworld=2, nconmax=128, njmax=256)
                with wp.ScopedCapture() as capture: mw.step(gm, gd)
                assert not np.any(gd.time.numpy())
                qp = gd.qpos.numpy(); qv = gd.qvel.numpy(); q[0] = qp; v[0] = qv
                for k in range(steps):
                    ctrl[:, motors] = controls[k]
                    for _ in range(substeps):
                        ctrl[0, springs] = replay.training_spring.gas_spring_force(qp[0, sq], qv[0, sv])
                        ctrl[1, springs] = 0.
                        gd.ctrl.assign(ctrl); wp.capture_launch(capture.graph)
                        qp = gd.qpos.numpy(); qv = gd.qvel.numpy()
                    q[k+1] = qp; v[k+1] = qv
                assert np.isfinite(q).all() and np.isfinite(v).all()
                assert np.max(abs(gd.time.numpy()-steps*.001)) < .03
            report['gpu_environment_steps'] += steps*substeps*2
            times = np.arange(steps+1)*.001
            # 独立运动学复查每个保存的1ms状态，不把重构姿态写回仿真。
            real = np.stack([np.interp(times, old['time_s'], old['qpos'][:, 0, j]) for j in range(model.nq)], axis=1)
            gap = []
            for branch, gas in enumerate(('on', 'off')):
                key = variant+'_gas_'+gas
                rows[key], g, a, ex = metrics(model, times, q[:, branch], real)
                gap.append(g.max(1)); np.savez_compressed(directory/(key+'-audit.npz'), time_s=times,
                    pin_gap_m=g, triangle_signed_area_m2=a, limit_excess_rad=ex)
            np.savez_compressed(directory/(variant+'-replay.npz'), time_s=times, qpos=q, qvel=v,
                                submitted_model_torque_nm=controls, max_pin_gap_m=np.asarray(gap).T)
            new_q.append(q[::10]); new_v.append(v[::10]); new_gaps.append(np.asarray(gap).T[::10])
            print(json.dumps(dict(session=session, variant=variant, results={k:rows[k] for k in rows if k.startswith(variant)})), flush=True)
        (directory/'report.json').write_text(json.dumps(rows, indent=2)+'\n')
        report['sessions'][session] = rows
    assert all(sha(Path(p)) == value for p, value in protected.items())
    report['response_fit_accepted'] = False
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copy2(__file__, args.output/Path(__file__).name)


if __name__ == '__main__': main()
