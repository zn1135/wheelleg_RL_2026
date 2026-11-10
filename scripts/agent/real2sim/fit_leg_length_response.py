"""Fit measured left/right leg length with ordered GPU free torque replay."""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def lengths(angles):
    result = []
    for side in (0, 1):
        pair = angles[..., 2*side:2*side+2]
        flat = pair.reshape(-1, 2)
        result.append(fit.prior.batch_leg_state(flat, np.zeros_like(flat))[0].reshape(pair.shape[:-1]))
    return np.stack(result, axis=-1)


def leg_pose(angles):
    """H7 five-bar wheel position and tilt relative to the body vertical."""
    points = []
    for side in (0, 1):
        pair = angles[..., 2*side:2*side+2]+np.array([np.pi, 0.])
        ab = .21*np.stack((np.cos(pair), np.sin(pair)), axis=-1)
        delta = ab[..., 1, :]-ab[..., 0, :]
        distance = np.linalg.norm(delta, axis=-1)
        normal = np.stack((-delta[..., 1], delta[..., 0]), axis=-1)/distance[..., None]
        points.append(ab.mean(axis=-2)+np.sqrt(.25**2-distance**2/4)[..., None]*normal)
    point = np.stack(points, axis=-2)
    tilt = np.pi/2-np.arctan2(point[..., 1], point[..., 0])
    return np.linalg.norm(point, axis=-1), np.arctan2(np.sin(tilt), np.cos(tilt)), point


def pose_metrics(directory, inputs):
    rows, trace = metrics(directory, inputs)
    model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
    mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
    with np.load(directory/'trials.npz') as saved:
        _, rt, rp = leg_pose(saved['real_motor_q'])
        _, st, sp = leg_pose(saved['qpos'][:, :, mq]*inputs['sign'][:4]+inputs['zero'])
    angle_error = np.degrees(np.arctan2(np.sin(st-rt[:, None]), np.cos(st-rt[:, None])))
    position_error = np.linalg.norm(sp-rp[:, None], axis=-1)*1000
    count = len(rows)
    for i, row in enumerate(rows):
        for branch, index in [('recorded_torque', i), ('recorded_target', count+i)]:
            e = angle_error[:, index]; p = position_error[:, index]
            row[branch].update(tilt_rmse_deg=np.sqrt(np.mean(e*e, axis=0)).tolist(),
                tilt_peak_deg=abs(e).max(0).tolist(), tilt_terminal_deg=abs(e[-1]).tolist(),
                endpoint_rmse_mm=np.sqrt(np.mean(p*p, axis=0)).tolist(),
                endpoint_peak_mm=p.max(0).tolist(), endpoint_terminal_mm=p[-1].tolist())
    trace.update(measured_tilt_rad=rt, simulated_tilt_rad=st,
                 measured_endpoint_m=rp, simulated_endpoint_m=sp)
    return rows, trace


def pose_score(trials, side=None):
    values = []
    for trial in trials:
        row = trial['recorded_torque']
        if not row['allowed']: return 1e6
        values.extend(row['endpoint_rmse_mm'] if side is None else [row['endpoint_rmse_mm'][side]])
    return float(np.sqrt(np.mean(np.square(values))))


def metrics(directory, inputs):
    report = json.loads((directory/'report.json').read_text())
    assert report['solver_profile'] == 'ordered' and report['numerical_regularization'] == 'original'
    assert fit.audit.digest(directory/'model.xml') == report['model_sha256']
    for path, digest in report['input_sha256'].items(): assert fit.audit.digest(Path(path)) == digest
    assert fit.audit.digest(Path(report['rc_path'])) == report['rc_sha256']
    model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
    mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
    with np.load(directory/'trials.npz') as saved:
        time = saved['time_s'].copy()
        real = lengths(saved['real_motor_q'])
        predicted = lengths(saved['qpos'][:, :, mq]*inputs['sign'][:4]+inputs['zero'])
    error = 1000.*(predicted-real[:, None])
    count = len(report['results']); rows = []
    for i, result in enumerate(report['results']):
        row = {}
        for branch, index in [('recorded_torque', i), ('recorded_target', count+i)]:
            e = error[:, index]
            physical = result[branch]
            row[branch] = dict(rmse_mm=np.sqrt(np.mean(e*e, axis=0)).tolist(),
                peak_mm=abs(e).max(0).tolist(), terminal_mm=abs(e[-1]).tolist(),
                allowed=bool(np.isfinite(e).all() and physical['original_assembly_branch']
                             and physical['limit_excess_deg'] <= .1),
                joint_rmse_deg=physical['rmse_deg'])
        rows.append(row)
    return rows, dict(time_s=time, measured_length_m=real, simulated_length_m=predicted)


def score(trials, side=None):
    # 实录力矩驱动自由响应；同目标反馈不参与拟合，以免闭环补偿掩盖模型偏差。
    values = []
    for trial in trials:
        row = trial['recorded_torque']
        if not row['allowed']: return 1e6
        values.extend(row['rmse_mm'] if side is None else [row['rmse_mm'][side]])
    return float(np.sqrt(np.mean(np.square(values))))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--population', type=int, default=24)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--sidewise', action='store_true')
    parser.add_argument('--extra-seeds', type=Path, nargs='+')
    parser.add_argument('--pose', action='store_true', help='Jointly fit length and leg tilt using wheel position error')
    args = parser.parse_args(); assert 8 <= args.population <= 32 and 0 <= args.generations <= 6
    args.output.mkdir(parents=True, exist_ok=False)
    measure = pose_metrics if args.pose else metrics
    objective = pose_score if args.pose else score
    fit.wp.init(); rng = np.random.default_rng(2026110405)
    protected_root = Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway')
    protected = fit.audit.protected_sources(protected_root, [])
    inputs = {name:fit.input_data(name) for name in fit.REFERENCES}
    original_pool = {}; cached = {}; source_hashes = {}; total = 0
    for name in ('original', 'wide'):
        directory = args.pool/'pool'/name
        original_pool[name] = json.loads((directory/'report.json').read_text())
        cached[name], _ = measure(directory, inputs[name])
        source_hashes[str(directory/'report.json')] = fit.audit.digest(directory/'report.json')
    profiles = [x['parameters'] for x in original_pool['original']['results']]
    assert profiles == [x['parameters'] for x in original_pool['wide']['results']]
    for extra in args.extra_seeds or []:
        for directory in sorted(extra.glob('generation-*')):
            if not (directory/'length-scores.json').exists(): continue
            profiles.extend(json.loads((directory/'parameters.json').read_text()))
            for name in ('original', 'wide'):
                rows, _ = measure(directory/name, inputs[name]); cached[name].extend(rows)
                source_hashes[str(directory/name/'report.json')] = fit.audit.digest(directory/name/'report.json')
        if (extra/'report.json').exists():
            profiles.append(json.loads((extra/'report.json').read_text())['selected_parameters'])
            for name in ('original', 'wide'):
                rows, _ = measure(extra/('final-'+name), inputs[name]); cached[name].extend(rows)
                source_hashes[str(extra/('final-'+name)/'report.json')] = fit.audit.digest(extra/('final-'+name)/'report.json')
    pool_scores = np.array([objective([cached[n][i] for n in ('original', 'wide')]) for i in range(len(profiles))])
    ranking = np.argsort(pool_scores)
    fit.search.write_json(args.output/'reused-pool-scores.json', dict(scores_mm=pool_scores.tolist(),
        best_index=int(ranking[0]), best_score_mm=float(pool_scores[ranking[0]]), source_report_hashes=source_hashes))
    side_ranking = [np.argsort([objective([cached[n][i] for n in ('original', 'wide')], side)
                               for i in range(len(profiles))]) for side in (0, 1)]
    vectors = np.empty((args.population, 2, 14))
    for i in range(args.population):
        vectors[i] = [fit.encode(profiles[(side_ranking[side] if args.sidewise else ranking)[i % 8]], side)
                      for side in (0, 1)]
        if i >= 8: vectors[i] = np.clip(vectors[i]+rng.normal(0., .07, (2, 14)), 0., 1.)
    history = []; evaluations = []

    def evaluate(values, generation):
        nonlocal total
        profiles = [fit.decode(v) for v in values]; rows = {}
        directory = args.output/f'generation-{generation:02d}'
        directory.mkdir(); fit.search.write_json(directory/'parameters.json', profiles)
        for name in ('original', 'wide'):
            _, report = fit.run(profiles, inputs[name], directory/name, solver_profile='ordered')
            total += report['new_gpu_environment_steps']
            rows[name], _ = measure(directory/name, inputs[name])
        scores = np.array([[objective([rows[n][i] for n in ('original', 'wide')], side) for side in (0, 1)]
                           if args.sidewise else objective([rows[n][i] for n in ('original', 'wide')])
                           for i in range(len(profiles))])
        fit.search.write_json(directory/'length-scores.json', dict(scores_mm=scores.tolist(), sessions=rows))
        evaluations.extend(dict(parameters=p, score_mm=float(np.sqrt(np.mean(np.square(s)))), generation=generation, index=i)
                           for i, (p, s) in enumerate(zip(profiles, scores)))
        return scores

    scores = evaluate(vectors, 0)
    for generation in range(args.generations+1):
        if generation:
            proposals = vectors.copy()
            best = np.stack([vectors[np.argmin(scores[:, s]), s] for s in (0, 1)]) if args.sidewise else vectors[np.argmin(scores)]
            for i in range(args.population):
                a, b = rng.choice(np.delete(np.arange(args.population), i), 2, replace=False)
                mutant = vectors[i]+.6*(best-vectors[i])+.65*(vectors[a]-vectors[b])
                if i < 4: mutant = best+rng.normal(0., .045/(generation**.5), (2, 14))
                mask = rng.uniform(size=(2, 14)) < .75
                proposals[i] = np.clip(np.where(mask, mutant, vectors[i]), 0., 1.)
            proposed = evaluate(proposals, generation)
            improve = proposed < scores; vectors[improve] = proposals[improve]; scores[improve] = proposed[improve]
        best_score = float(np.sqrt(np.mean(scores.min(0)**2))) if args.sidewise else float(scores.min())
        entry = dict(generation=generation, best_length_rms_mm=best_score, new_gpu_environment_steps=total)
        history.append(entry); fit.search.write_json(args.output/'progress.json', history)
        print(json.dumps(entry), flush=True)
    best_new = min(evaluations, key=lambda x:x['score_mm'])
    selected = profiles[int(ranking[0])]; selected_score = float(pool_scores[ranking[0]])
    if best_new['score_mm'] < selected_score: selected = best_new['parameters']; selected_score = best_new['score_mm']
    if args.sidewise:
        best_score = float(np.sqrt(np.mean(scores.min(0)**2)))
        if best_score < selected_score:
            selected = fit.decode(np.stack([vectors[np.argmin(scores[:, s]), s] for s in (0, 1)]))
            selected_score = best_score
    fit.search.write_json(args.output/'selected-parameters.json', selected)
    final = {}; length_results = {}; baseline = {}; baseroot = Path('artifacts/real2sim/h7-ordered-refit-20261104-02-best')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in fit.REFERENCES:
        directory = args.output/('final-'+name)
        _, result = fit.run([selected], inputs[name], directory, full=True, solver_profile='ordered')
        total += result['new_gpu_environment_steps']; final[name] = result['results'][0]
        rows, trace = measure(directory, inputs[name]); length_results[name] = rows[0]
        baseline_rows, old = measure(baseroot/('final-'+name), inputs[name]); baseline[name] = baseline_rows[0]
        np.savez_compressed(directory/'leg-length.npz', **trace)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for side, ax in enumerate(axes):
            ax.plot(trace['time_s'], trace['measured_length_m'][:, side]*1000, label='Measured')
            ax.plot(trace['time_s'], trace['simulated_length_m'][:, 0, side]*1000, label='Length fit / recorded torque')
            ax.plot(trace['time_s'], trace['simulated_length_m'][:, 1, side]*1000, label='Length fit / target feedback', alpha=.6)
            ax.plot(old['time_s'], old['simulated_length_m'][:, 0, side]*1000, label='Previous angle fit', alpha=.5)
            ax.set_ylabel(('Left' if side == 0 else 'Right')+' leg length (mm)'); ax.grid(alpha=.25)
        axes[0].legend(fontsize=8); axes[-1].set_xlabel('Time (s)'); fig.tight_layout()
        fig.savefig(args.output/(name+'-leg-length.png'), dpi=150); plt.close(fig)
        if args.pose:
            fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
            for side in (0, 1):
                for column, (key, scale, unit) in enumerate([('length_m', 1000., 'length (mm)'), ('tilt_rad', 180./np.pi, 'tilt (deg)')]):
                    ax = axes[side, column]
                    ax.plot(trace['time_s'], trace['measured_'+key][:, side]*scale, label='Measured')
                    ax.plot(trace['time_s'], trace['simulated_'+key][:, 0, side]*scale, label='Same torque')
                    ax.plot(trace['time_s'], trace['simulated_'+key][:, 1, side]*scale, label='Same target', alpha=.7)
                    ax.set_ylabel(('Left ' if side == 0 else 'Right ')+unit); ax.grid(alpha=.25)
            axes[0, 0].legend(fontsize=8)
            for ax in axes[-1]: ax.set_xlabel('Time (s)')
            fig.tight_layout(); fig.savefig(args.output/(name+'-leg-pose.png'), dpi=150); plt.close(fig)
    assert fit.audit.protected_sources(protected_root, []) == protected
    fit.search.write_json(args.output/'report.json', dict(selected_parameters=selected, final=final,
        length_results=length_results, previous_angle_fit_length_results=baseline, history=history,
        objective='Minimum equal-weight leg length RMS over original/wide, left/right, recorded torque only; feedback and joint angles separately reported',
        combined_fit_length_rms_mm=score([length_results[n] for n in ('original', 'wide')]),
        selection_sampled_length_rms_mm=selected_score, selection_sample_dt_s=.01,
        sidewise_search=args.sidewise, side_composition_requires_fixed_base=True,
        source_pool_report_hashes=source_hashes,
        candidates_reused=len(profiles), candidates_new=len(evaluations), new_gpu_environment_steps=total,
        protected_source_checks=protected, solver_profile='ordered', shared_model_modified=False,
        hardware_access=False, policy_training_updates=0, accepted=False,
        development_sessions=['original', 'wide'], regression_sessions=['slow', 'fast'],
        physical_parameters_uniquely_identified=False, future_measured_position_feedback=False))
    if args.pose:
        path = args.output/'report.json'; report = json.loads(path.read_text())
        report.update(pose_results=length_results,
            objective='Minimum wheel endpoint RMS in the H7 leg plane, jointly constraining length and tilt; same torque, original/wide and left/right equally weighted',
            tilt_definition='wrap(pi/2-atan2(OP_y, OP_x)); body vertical zero, H7 leg plane, no world pitch subtraction',
            combined_fit_endpoint_rms_mm=objective([length_results[n] for n in ('original', 'wide')]),
            selection_sampled_endpoint_rms_mm=selected_score)
        fit.search.write_json(path, report)
    from scripts.agent.real2sim import ordered_leg_constraints
    for source in (Path(__file__), Path(fit.__file__), Path(ordered_leg_constraints.__file__)):
        shutil.copy2(source, args.output/source.name)
    print(json.dumps(dict(completed=str(args.output), length_results=length_results)), flush=True)


if __name__ == '__main__': main()
