"""Verify a finite passive H7_ros capture; never opens hardware or runs a policy.

Uses the project's CRC/sequence parsers. A fixed settling interval is explicit;
missing SDK/PD activity remains unmeasured, never a successful active test.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from h7_output_capture import CaptureParser
from h7_latency_capture import LatencyParser


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(count=len(values), min=float(values.min()), median=float(np.median(values)),
                max=float(values.max()), p99=float(np.percentile(values, 99))) if len(values) else dict(count=0)


def gaps(values):
    return sum((b-a) % 2**32 != 1 for a, b in zip(values, values[1:]))


def classify_tail_candidates(session, start_s, end_s):
    """Attribute payload tails only to complete packets accepted by real CRC parsers."""
    raw = bytearray()
    ends, times = [], []
    for event in mirror.iter_session_events(session):
        if event['type'] == 'data':
            raw.extend(bytes.fromhex(event['data_hex']))
            ends.append(len(raw)); times.append(event['t_ns']/1e9)
    spans=[]
    for magic, decoder, record_size in ((b'H7OUT001',CaptureParser,88),(b'H7LAT001',LatencyParser,64)):
        offset=0
        while True:
            offset=raw.find(magic,offset)
            if offset<0: break
            if offset+40<=len(raw):
                count=int.from_bytes(raw[offset+12:offset+14],'little')
                stop=offset+40+count*record_size+2
                if count<=128 and stop<=len(raw):
                    parser=decoder()
                    packets=parser.feed(bytes(raw[offset:stop]))
                    if len(packets)==1 and not parser.rejected:
                        spans.append((offset,stop,magic.decode(),packets[0]['sequence']))
            offset+=1
    tails=[]; offset=0
    while True:
        offset=raw.find(mirror.TAIL,offset)
        if offset<0: break
        stop=offset+4
        frame=None
        for size in (644,196):
            if stop<size: continue
            try: frame=mirror.decode_frame(bytes(raw[stop-size:stop]))
            except ValueError: continue
            else:
                spans.append((stop-size,stop,'mirror',mirror.read_u32(frame[6:8])))
                break
        at=times[bisect_right(ends,stop-1)]
        if start_s<=at<=end_s and frame is None:
            tails.append((offset,stop,at))
        offset+=1
    spans.sort(); starts=[x[0] for x in spans]
    payload,unexplained=[],[]
    for begin,stop,at in tails:
        index=bisect_right(starts,begin)-1
        container=spans[index] if index>=0 else None
        item=dict(byte_offset=begin,receive_s=at)
        if container and stop<=container[1]:
            payload.append(item|dict(crc_valid_packet=container[2],packet_sequence=container[3]))
        else: unexplained.append(item)
    return dict(payload_markers=payload,unexplained_markers=unexplained,valid_packet_spans=len(spans))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--settle-seconds', type=float, default=3.)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.seconds > 0 and args.settle_seconds >= 0
    meta, _, _ = mirror.load_session(args.session)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert not args.out.exists()
    frame_parser, output_parser, latency_parser = mirror.FrameParser(), CaptureParser(), LatencyParser()
    frames, receive_s, packets, latency, connections = [], [], [], [], 0
    rejection_base = None
    for event in mirror.iter_session_events(args.session):
        at = event['t_ns'] / 1e9
        if event['type'] == 'connected':
            connections += 1
        if event['type'] != 'data':
            continue
        if at > args.settle_seconds + args.seconds:
            break
        if at >= args.settle_seconds and rejection_base is None:
            rejection_base = (frame_parser.rejected, output_parser.rejected, latency_parser.rejected)
        chunk = bytes.fromhex(event['data_hex'])
        current_frames = frame_parser.feed(chunk)
        current_output = output_parser.feed(chunk)
        current_latency = latency_parser.feed(chunk)
        if at < args.settle_seconds:
            continue
        frames.extend(current_frames)
        receive_s.extend([at]*len(current_frames))
        packets.extend(current_output)
        latency.extend(current_latency)
    assert frames and packets and latency, 'missing required telemetry/output/latency stream'
    f = np.asarray(frames)
    controls = [r for p in packets for r in p['records'] if r[1] == 1]
    all_output = [r for p in packets for r in p['records']]
    timing = [r for p in latency for r in p['records']]
    assert controls and timing
    requests = np.array([r[6:12] for r in controls])
    kinds = Counter(r[1] for r in timing)
    policy = [r for r in timing if r[1] == 2]
    actuation = [r for r in timing if r[1] == 4]
    network_calls = [r for r in policy if r[11] != 0xffffffff]
    pd = [r for r in timing if r[1] == 3 and r[3] & 1]
    rejections = dict(zip(('telemetry', 'output', 'latency'),
                         (a-b for a,b in zip((frame_parser.rejected, output_parser.rejected, latency_parser.rejected), rejection_base))))
    boundaries = dict(output_packet_gaps=gaps([p['sequence'] for p in packets]),
                      output_record_gaps=gaps([r[0] for r in all_output]),
                      output_new_queue_drops=packets[-1]['dropped']-packets[0]['dropped'],
                      latency_packet_gaps=gaps([p['sequence'] for p in latency]),
                      latency_record_gaps=gaps([r[0] for r in timing]),
                      latency_new_queue_drops=latency[-1]['dropped']-latency[0]['dropped'],
                      telemetry_sequence_gaps=gaps([mirror.read_u32(row[6:8]) for row in frames]))
    tail_attribution=classify_tail_candidates(args.session,args.settle_seconds,args.settle_seconds+args.seconds)
    checks = dict(recording_saved_complete=bool(meta['complete']), exactly_one_connection=connections==1,
                  sufficient_duration=receive_s[-1]-receive_s[0]>=args.seconds-.5,
                  all_wire_v3=bool(np.all(f[:,4]==3)), no_actual_packet_corruption=not (rejections['output'] or rejections['latency'] or tail_attribution['unexplained_markers']),
                  contiguous_observed_span=not any(boundaries.values()),
                  fault_free=bool(np.all(f[:,2]==0) and all(r[5]==0 for r in controls)),
                  network_initialized=bool(np.all(f[:,155]==1)),
                  motor_and_controllers_disabled=bool(np.all((f[:,1].astype(int) & 0x3f1)==0)),
                  rc_not_engaged=bool(np.all((f[:,153].astype(int)&6)==0)),
                  output_snapshots_present=bool(np.all((f[:,47].astype(int)&2)!=0)),
                  zero_latest_requests=bool(np.isfinite(f[:,132:138]).all() and not np.any(f[:,132:138])),
                  zero_all_control_requests=bool(np.isfinite(requests).all() and not np.any(requests)))
    ages = dict(imu_ms=stats(f[f[:,144]>=0,144]), rc_ms=stats(f[f[:,145]>=0,145]))
    ages.update({name+'_ms': stats(f[f[:,138+i]>=0,138+i]) for i,name in enumerate(mirror.MOTORS)})
    periods = dict(policy_task_entry_us=stats(np.diff([r[4] for r in policy])/1000),
                   base_actuation_entry_us=stats(np.diff([r[4] for r in actuation])/1000),
                   idle_policy_task_duration_us=stats([(r[5]-r[4])/1000 for r in policy]),
                   base_actuation_duration_us=stats([(r[5]-r[4])/1000 for r in actuation]))
    report = dict(cold_check_pass=all(checks.values()), scope='passive disabled-state check only; no motor commands, no policy stimulation or physical acceptance',
                  session=str(args.session), stream_sha256=meta['stream_sha256'], settle_seconds=args.settle_seconds,
                  requested_steady_seconds=args.seconds, observed_seconds=receive_s[-1]-receive_s[0],
                  telemetry_frames=len(frames), control_records=len(controls), timing_records=len(timing),
                  checks=checks, stream_rejections=rejections, rejected_tail_attribution=tail_attribution, observed_boundaries=boundaries,
                  max_abs_request_nm=float(np.abs(requests).max()),
                  faults=dict(Counter(int(x) for x in f[:,2])), rc_strategy=dict(Counter(int(x) for x in f[:,152])),
                  recovery_phase=dict(Counter(int(x) for x in f[:,154])), feedback_ages=ages, observed_periods=periods,
                  active_sdk_calls=len(network_calls), active_pd_records=len(pd),
                  on_board_policy_numerical_comparison='unmeasured: disabled state produces no inference pairs',
                  network_worst_duration='unmeasured' if not network_calls else stats([r[11] for r in network_calls]),
                  active_500hz_pd='unmeasured' if not pd else 'requires separate active analysis',
                  motion_acceptance='not performed; existing GPU result remains 11/987 failed',
                  protocol_note='v3 has 32 physical telemetry slots; this firmware uses 25 obs plus 7 padding slots; 125 history values are not transmitted',
                  limitations=['settling interval is excluded by the declared time cutoff; full raw session remains preserved',
                               'observed scheduling periods in the disabled state are not active inference/control worst-case bounds',
                               'expected firmware manifest is operator provenance, not firmware self-attestation',
                               'zero requested torque is not an independent shaft torque or dynamics measurement'],
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'sim2sim/real2sim_huananhu.py',ROOT/'sim2sim/h7_output_capture.py',ROOT/'sim2sim/h7_latency_capture.py')})
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('cold_check_pass','telemetry_frames','control_records','checks','observed_periods','active_sdk_calls','active_pd_records')},indent=2))
    if not report['cold_check_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
