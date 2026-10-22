#!/usr/bin/env python3
"""Offline fault chronology from a hashed H7 recording; never opens a port.

Select by recorded receive time, retain board timestamps, and exclude older
policy/output snapshots from numeric checks. No dynamics or hardware changes.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from h7_output_capture import CaptureParser


def identity(path):
    return {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--comparison-checkpoint', type=Path)
    args = p.parse_args()
    meta, xml, config = mirror.load_session(args.session)
    args.out.mkdir(parents=True, exist_ok=False)
    parser, clock, capture = mirror.FrameParser(), mirror.FrameClock(), CaptureParser()
    frames, times, records, packets = [], [], [], []
    for event in mirror.iter_session_events(args.session):
        at = event['t_ns'] / 1e9
        if at > args.seconds:
            break
        if event['type'] != 'data':
            if event['type'] == 'connected':
                parser.buffer.clear()
                capture.buffer.clear()
            continue
        chunk = bytes.fromhex(event['data_hex'])
        for f in parser.feed(chunk):
            if clock.accept(f):
                frames.append(f)
                times.append(at)
        for packet in capture.feed(chunk):
            packets.append(packet)
            records.extend((at, row) for row in packet['records'])
    f = np.asarray(frames)
    t = np.asarray(times)
    ticks = np.asarray([mirror.read_u32(row[8:10]) for row in f])
    board_s = ticks / 1000
    pd = ((f[:, 47].astype(int) & 2) != 0) & ((f[:, 109].astype(int) & 1) != 0)
    permitted = (f[:, 109].astype(int) & 32) != 0
    rl = f[:, 152] == 0
    active = rl & pd & permitted
    rows = []
    for at, frame in zip(t, f):
        row = dict(receive_s=float(at), board_tick_ms=mirror.read_u32(frame[8:10]),
                   mirror_sequence=mirror.read_u32(frame[6:8]), online_mask=int(frame[0]),
                   fault=int(frame[2]), state_bits=int(frame[1]), source_valid_mask=int(frame[29]))
        row.update(mirror.telemetry_row(frame))
        rows.append(row)
    with (args.out / 'telemetry.csv').open('w') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.out / 'frames.npz', frames=f, receive_s=t, board_tick_ms=ticks)

    transitions, last = [], None
    for i, frame in enumerate(f):
        key = tuple(int(frame[j]) for j in (152, 2, 109, 154, 0))
        if key != last:
            transitions.append(dict(receive_s=float(t[i]), board_tick_ms=int(ticks[i]),
                                    strategy=key[0], fault=key[1], output_flags=key[2],
                                    recovery=key[3], online_mask=key[4],
                                    motor_source_ages_ms=frame[138:144].tolist()))
        last = key
    episodes = []
    for start in np.flatnonzero(active & ~np.r_[False, active[:-1]]):
        end = start
        while end + 1 < len(f) and active[end + 1]:
            end += 1
        following = next((j for j in range(end + 1, len(f)) if int(f[j, 2]) & 4), None)
        episodes.append(dict(start_receive_s=float(t[start]), last_active_receive_s=float(t[end]),
                             start_board_tick_ms=int(ticks[start]),
                             first_motor_fault_receive_s=float(t[following]) if following else None,
                             active_to_motor_fault_ms=int(ticks[following] - ticks[start]) if following else None,
                             initial_model_pos_rad=f[start, 122:126].tolist(),
                             initial_target=f[start, 110:116].tolist(),
                             initial_firmware_torque_nm=f[start, 132:138].tolist(),
                             fault_motor_ages_ms=f[following, 138:144].tolist() if following else None))

    # Source config, rather than hand-entered gains or a different XML version.
    text = config.read_text().split('[MACHINE_ID_SMALL_WHEELLEG]')[0]
    def values(name):
        body = re.search(r'\.' + name + r'\s*=\s*\{([^}]+)\}', text).group(1)
        return np.asarray([float(x.strip().rstrip('f')) for x in body.split(',')], dtype=np.float32)
    def scalar(name):
        return np.float32(re.search(r'\.' + name + r'\s*=\s*([-\d.]+)f', text).group(1))
    nominal, reference, signs = values('nominal_pos'), values('reference_pos'), values('motor_sign')
    used = f[active, 116:122]
    order = [0, 1, 3, 4, 2, 5]
    target = np.empty((len(used), 6), dtype=np.float32)
    target[:, :4] = np.clip(nominal + scalar('pos_scale') * used[:, order[:4]],
                           reference - scalar('position_limit'), reference + scalar('position_limit'))
    target[:, 4:] = np.clip(scalar('wheel_vel_scale') * used[:, order[4:]],
                           -scalar('wheel_speed_limit'), scalar('wheel_speed_limit'))
    torque = np.empty_like(target)
    torque[:, :4] = np.clip(values('p_gains') * (target[:, :4] - f[active, 122:126])
                            - values('d_gains') * f[active, 126:130],
                            -scalar('leg_torque_limit'), scalar('leg_torque_limit'))
    torque[:, 4:] = np.clip(values('wheel_gain') * (target[:, 4:] - f[active, 130:132]),
                            -scalar('wheel_torque_limit'), scalar('wheel_torque_limit'))
    torque *= signs
    pd_check = dict(samples=int(active.sum()),
                    max_target_error=float(np.max(np.abs(target - f[active, 110:116]))) if len(used) else None,
                    max_torque_error_nm=float(np.max(np.abs(torque - f[active, 132:138]))) if len(used) else None,
                    scope='Same recorded action/PD feedback, target and gated-request arithmetic only; not measured shaft torque.')

    fresh = {}
    for frame in f[rl]:
        if int(frame[47]) & 1 and mirror.read_u32(frame[50:52]) >= ticks[0]:
            fresh[mirror.read_u32(frame[48:50])] = frame
    numeric = None
    if args.checkpoint and fresh:
        import torch
        sys.path.insert(0, str(ROOT / 'mjlab_training/src'))
        from ct_mjlab.scripts.export import load_actor
        selected = np.unique(np.linspace(0, len(fresh) - 1, min(64, len(fresh))).astype(int))
        sample = np.asarray(list(fresh.values()))[selected]
        obs = torch.from_numpy(sample[:, 57:96].copy())
        recorded = sample[:, 96:102]
        numeric = dict(samples=len(selected), expected=identity(args.checkpoint))
        with torch.inference_mode():
            predicted = load_actor(args.checkpoint)(obs).numpy()
            numeric['max_action_error'] = float(np.max(np.abs(predicted - recorded)))
            numeric['rmse_action_error'] = float(np.sqrt(np.mean((predicted - recorded) ** 2)))
            if args.comparison_checkpoint:
                old = load_actor(args.comparison_checkpoint)(obs).numpy()
                numeric['comparison'] = identity(args.comparison_checkpoint)
                numeric['comparison_max_action_error'] = float(np.max(np.abs(old - recorded)))
                numeric['comparison_rmse_action_error'] = float(np.sqrt(np.mean((old - recorded) ** 2)))
        np.savez_compressed(args.out / 'policy-check.npz', obs=sample[:, 57:96], recorded=recorded,
                            predicted=predicted, policy_sequence=sample[:, 48:50])

    # The USB open can drain old output packets. Keep them in raw data, but never
    # associate those earlier board timestamps with the current RL incident.
    current = [(at, row) for at, row in records if ticks[0] * 1e6 <= row[2] <= ticks[-1] * 1e6]
    controls = [(at, row) for at, row in current if row[1] == 1]
    can = [(at, row) for at, row in current if row[1] == 2]
    output_sequences = [row[0] for _, row in current]
    output_gaps = [dict(after=a, before=b) for a, b in zip(output_sequences, output_sequences[1:])
                   if (b - a) % 2**32 != 1]
    high_rate_episodes, running = [], None
    for _, row in controls:
        nonzero = row[21] == 0 and any(row[6:12])
        if nonzero:
            if running is None:
                running = dict(start_board_ns=row[2], last_nonzero_board_ns=row[2], records=0)
                high_rate_episodes.append(running)
            running['last_nonzero_board_ns'] = row[2]
            running['records'] += 1
        elif running is not None:
            running['first_zero_board_ns'] = row[2]
            running['duration_to_first_zero_ms'] = (row[2] - running['start_board_ns']) / 1e6
            running = None
    # Infer only the driver's integer representation; do not call this a
    # measured physical zero, and never feed the reduced angles to hardware.
    span = 2 * scalar('dm_pos_max')
    initial_turns = np.floor((f[0, 10:14] - values('dm_zero') + span / 2) / span).astype(int)
    report = dict(session=identity(args.session / 'meta.json'), stream_sha256=meta['stream_sha256'],
                  seconds=args.seconds, model=identity(xml), config=identity(config),
                  analyzer=identity(Path(__file__)), mirror_frames=len(f),
                  receive_range_s=[float(t[0]), float(t[-1])],
                  strategy_counts=dict(collections.Counter(str(int(x)) for x in f[:, 152])),
                  fault_counts=dict(collections.Counter(str(int(x)) for x in f[:, 2])),
                  missing_submitted_frames=clock.missing, restarts=clock.restarts,
                  rejected_frame_candidates=parser.rejected, fresh_RL_policy_pairs=len(fresh),
                  RL_PD_permitted_frames=int(active.sum()), episodes=episodes, transitions=transitions,
                  PD_numeric_check=pd_check, policy_numeric_check=numeric,
                  current_high_rate_control_records=len(controls), current_CAN_records=len(can),
                  current_high_rate_event_sequence_gaps=output_gaps,
                  high_rate_nonzero_RL_episodes=high_rate_episodes,
                  initial_firmware_continuous_angles_rad=f[0, 10:14].tolist(),
                  initial_model_continuous_angles_rad=(f[0, 10:14] * signs[:4]).tolist(),
                  inferred_initial_driver_turn_indices=initial_turns.tolist(),
                  equivalent_bounded_FW_angles_rad=(f[0, 10:14] - initial_turns * span).tolist(),
                  branch_scope='Representation diagnostic from driver span/zero only; not physical calibration or a correction.',
                  current_CAN_HAL_failures=sum(row[20] != 0 for _, row in can),
                  CRC_rejected_output_packets=capture.rejected,
                  excluded_older_or_outside_output_records=len(records) - len(current),
                  saturation_fraction_per_motor=(np.mean(np.isclose(np.abs(f[active, 132:138]),
                      np.array([40, 40, 40, 40, 3.9, 3.9]), atol=1e-5), axis=0).tolist()) if active.any() else None,
                  limits=['USB receive time is not control delay; high-rate data selected also by board time.',
                          'No measured shaft torque, motor error codes, voltage or physical translation.',
                          'The recorded XML is the Jetson display model, not a board policy identifier.',
                          'Offline arithmetic only; no controller, calibration, weights or model changes.'])
    save(args.out / 'report.json', report)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    labels = ['lf0', 'lf00', 'rf0', 'rf00']
    for j, label in enumerate(labels):
        valid = (f[:, 29].astype(int) & (1 << (j + 1))) != 0
        q = signs[j] * f[:, 10 + j]
        axes[0].plot(t, np.where(valid, q, np.nan), label=label)
        axes[2].plot(t, f[:, 138 + j], label=label)
        axes[1].plot(t, f[:, 132 + j], label=label)
    axes[0].set_ylabel('Model joint angle [rad]')
    axes[1].set_ylabel('Gated request [Nm]')
    axes[2].set_ylabel('Feedback age [ms]')
    axes[2].set_ylim(-2, 110)
    axes[2].axhline(50, color='k', ls='--', label='DM offline threshold')
    for j, label in enumerate(('left wheel', 'right wheel')):
        valid = (f[:, 29].astype(int) & (1 << (j + 5))) != 0
        axes[3].plot(t, np.where(valid, f[:, 20 + j] * signs[4 + j], np.nan), label=label)
    axes[3].set_ylabel('Wheel speed [rad/s]')
    axes[4].step(t, f[:, 2], label='fault mask', where='post')
    axes[4].step(t, f[:, 154], label='recovery phase', where='post')
    axes[4].step(t, f[:, 152], label='strategy (RL=0)', where='post')
    axes[4].set_ylabel('Firmware status')
    for ax in axes:
        for episode in episodes:
            ax.axvline(episode['start_receive_s'], color='g', alpha=.6, lw=.8)
            if episode['first_motor_fault_receive_s']:
                ax.axvline(episode['first_motor_fault_receive_s'], color='r', alpha=.6, lw=.8)
        ax.grid(alpha=.25)
        ax.legend(loc='upper right', ncol=4, fontsize=8)
    axes[-1].set_xlim(0, args.seconds)
    axes[-1].set_xlabel('Recorded USB receive time [s]; green: valid RL PD, red: motor fault')
    fig.suptitle(f'First {args.seconds:g} s of actual H7 recording: angles, requests and feedback status')
    fig.tight_layout()
    fig.savefig(args.out / 'first30.png', dpi=140)
    if episodes:
        axes[-1].set_xlim(8, 12.3)
        fig.savefig(args.out / 'fault-zoom.png', dpi=140)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ('mirror_frames', 'fresh_RL_policy_pairs',
                      'RL_PD_permitted_frames', 'PD_numeric_check', 'policy_numeric_check',
                      'current_CAN_HAL_failures')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
