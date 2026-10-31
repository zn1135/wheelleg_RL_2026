#!/usr/bin/env python3
"""Frozen zero-command pitch measurements and paired checkpoint progression; no training."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.agent.stand_v2 import evaluate_stand_v2_phase_mix as evaluation


FIELDS = ('pitch_signed_rad', 'roll_signed_rad', 'gravity_z', 'live', 'stood',
          'command_vx', 'command_yaw', 'height_m', 'base_force_N', 'left_support_N', 'right_support_N')
BASE_RECORDER = evaluation.Recorder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PitchRecorder(BASE_RECORDER):
    def __init__(self, raw, **kwargs):
        super().__init__(raw, record_physical=False)
        self.pitch_trace = torch.empty((20000, raw.num_envs, len(FIELDS)), device=raw.device)

    def sample(self):
        super().sample()
        raw = self.raw
        q = raw.sim.data.qpos[:, self.root_q]
        w, x, y, z = q[:, 3:].unbind(-1)
        gx, gy, gz = 2*(w*y-x*z), -2*(y*z+w*x), -1+2*(x*x+y*y)
        state = raw.command_manager.get_term('standup')
        support = -raw.scene['wheel_contact'].data.force[:, :, 2]
        base = raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
        self.pitch_trace[self.count-1] = torch.stack((gx.clamp(-1., 1.).asin(),
            torch.atan2(-gy, -gz), gz, self.live.float(), state.has_stood.float(),
            state.command[:, 0], state.command[:, 1], q[:, 2]-raw.scene.env_origins[:, 2],
            base, *support.unbind(-1)), -1)


def measure(args):
    recorders = []

    def recorder(raw, **kwargs):
        value = PitchRecorder(raw, **kwargs)
        recorders.append(value)
        return value

    # 复用原40秒/500Hz首回合评估；严格加载原契约后，统一关闭速度与航向外环。
    with patch.object(evaluation, 'Recorder', recorder), patch.object(
            torch.optim.Adam, 'step', side_effect=AssertionError('Frozen pitch measurement')):
        evaluation.main(argparse.Namespace(checkpoint=args.checkpoint, output=args.output,
            conditions='nominal', seed=args.seed, num_envs=args.num_envs,
            device=args.device, record_physical=False))
    r = recorders[0]
    assert r.count == 20000
    values = r.pitch_trace.cpu().numpy()
    assert np.isfinite(values).all()
    assert np.count_nonzero(values[:, :, 5:7]) == 0
    report_path = args.output/'result.json'
    report = json.loads(report_path.read_text())
    report.update(pitch_fields=FIELDS, signed_pitch_definition='asin(projected_gravity_body_x), target zero',
        measured_vx_and_yaw_commands_identically_zero=True, diagnostic_source_sha256=digest(__file__))
    with np.load(args.output/'episodes.npz') as episodes:
        # 仅筛选确实站起、双轮支撑、机身离地的样本；角速度单列，避免与pitch精度混为一项。
        tail = values[-500:]
        weight = r.weight.cpu().numpy()
        support = tail[:, :, 9:11]
        standing = (episodes['ever_stood'] & episodes['survived'] & episodes['finite']
            & (tail[:, :, 3] > .5).all(0) & (tail[:, :, 2] < 0.).all(0)
            & (tail[:, :, 8] <= 1.).all(0)
            & (support >= np.maximum(.1*weight, 5.)[None, :, None]).all((0, 2))
            & ((support.sum(-1) >= .8*weight) & (support.sum(-1) <= 1.2*weight)).all(0))
        rows = []
        for i in range(args.num_envs):
            signed = np.rad2deg(tail[:, i, 0])
            rows.append(dict(env=i, standing_for_pitch_comparison=bool(standing[i]),
                ever_stood=bool(episodes['ever_stood'][i]), balance_passed=bool(episodes['passed'][i]),
                final1_mean_abs_pitch_deg=float(np.abs(signed).mean()),
                final1_mean_signed_pitch_deg=float(signed.mean()),
                final1_peak_abs_pitch_deg=float(np.abs(signed).max()),
                final5_mean_abs_pitch_deg=float(np.rad2deg(np.abs(values[-2500:, i, 0])).mean())))
    report['pitch_episodes'] = rows
    report['standing_for_pitch_comparison'] = int(standing.sum())
    np.savez_compressed(args.output/'pitch.npz', values=values, fields=np.asarray(FIELDS),
        time_s=(np.arange(len(values))+1)*.002, standing=standing)
    (args.output/'pitch-source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'evaluation-source.py').write_bytes(Path(evaluation.__file__).read_bytes())
    report['pitch_sha256'] = digest(args.output/'pitch.npz')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    selected = [v['final1_mean_abs_pitch_deg'] for v in rows if v['standing_for_pitch_comparison']]
    print(json.dumps(dict(pitch_measurement_complete=True, standing=len(selected),
        median_abs_pitch_deg=float(np.median(selected)) if selected else None), ensure_ascii=False), flush=True)


def summarize(args):
    args.output.mkdir(parents=True, exist_ok=False)
    labels = [v[0] for v in args.measurement]
    paths = [Path(v[1]) for v in args.measurement]
    reports = [json.loads((p/'result.json').read_text()) for p in paths]
    reference = np.load(paths[0]/'initial.npz')
    initial_equal = []
    for path, report in zip(paths, reports):
        assert report['parameter_updates'] == 0 and report['actor_and_encoder_unchanged']
        assert report['measured_vx_and_yaw_commands_identically_zero']
        assert report['xml_sha256'] == reports[0]['xml_sha256']
        assert (report['seed'], report['num_envs']) == (reports[0]['seed'], reports[0]['num_envs'])
        with np.load(path/'initial.npz') as current:
            assert set(current.files) == set(reference.files)
            same = {k:bool(np.array_equal(current[k], reference[k])) for k in reference.files}
            assert all(same.values()), (str(path), same)
            initial_equal.append(same)
    standing = np.asarray([[e['standing_for_pitch_comparison'] for e in r['pitch_episodes']] for r in reports])
    common = standing.all(0)
    assert common.any(), 'No common standing samples; do not compare changing success sets'
    rows = []
    for label, path, report, mask in zip(labels, paths, reports, standing):
        values = np.asarray([v['final1_mean_abs_pitch_deg'] for v in report['pitch_episodes']])
        signed = np.asarray([v['final1_mean_signed_pitch_deg'] for v in report['pitch_episodes']])
        rows.append(dict(label=label, source_report=str((path/'result.json').resolve()),
            source_report_sha256=digest(path/'result.json'), checkpoint=report['checkpoint'],
            checkpoint_sha256=report['checkpoint_sha256'], num_envs=report['num_envs'],
            standing=int(mask.sum()), balance_passed=report['passed'],
            common_abs_pitch_median_deg=float(np.median(values[common])),
            common_abs_pitch_range_deg=[float(values[common].min()),float(values[common].max())],
            common_signed_pitch_median_deg=float(np.median(signed[common])),
            standing_abs_pitch_median_deg=float(np.median(values[mask])) if mask.any() else None))
    for i, row in enumerate(rows):
        row['delta_from_1999_deg'] = row['common_abs_pitch_median_deg']-rows[0]['common_abs_pitch_median_deg']
        row['delta_from_previous_deg'] = None if i == 0 else row['common_abs_pitch_median_deg']-rows[i-1]['common_abs_pitch_median_deg']
    report = dict(contract='stand-v2-zero-command-pitch-progression-v1', rows=rows,
        common_standing_env_ids=np.flatnonzero(common).tolist(), initial_arrays_equal=initial_equal,
        source_sha256=digest(__file__), parameter_updates=0,
        scope='One-seed nominal 40s/500Hz frozen zero-vx/zero-yaw measurements. Same physical initial arrays. '
              'Pitch target zero; compares the same standing samples across all checkpoints. '
              'No pitch pass threshold or causal claim; failed get-up counts remain explicit.')
    (args.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(11,4.5))
    y = [r['common_abs_pitch_median_deg'] for r in rows]
    axis.plot(np.arange(len(rows)),y,'o-')
    for i,v in enumerate(y):axis.annotate(f'{v:.2f}°',(i,v),xytext=(0,8),textcoords='offset points',ha='center')
    axis.set_xticks(np.arange(len(rows)),labels,rotation=20,ha='right')
    axis.set_ylabel('Median absolute pitch error (deg)')
    axis.set_title(f'Zero-command posture progression; seed {reports[0]["seed"]}; common n={int(common.sum())}')
    axis.grid(alpha=.25);fig.tight_layout();fig.savefig(args.output/'pitch-progression.png',dpi=150);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    one=commands.add_parser('measure')
    one.add_argument('--checkpoint',type=Path,required=True)
    one.add_argument('--output',type=Path,required=True)
    one.add_argument('--num-envs',type=int,default=32)
    one.add_argument('--seed',type=int,default=43)
    one.add_argument('--device',default='cuda:0')
    one.set_defaults(func=measure)
    many=commands.add_parser('summarize')
    many.add_argument('--measurement',nargs=2,action='append',required=True,metavar=('LABEL','DIRECTORY'))
    many.add_argument('--output',type=Path,required=True)
    many.set_defaults(func=summarize)
    args=parser.parse_args();args.func(args)
