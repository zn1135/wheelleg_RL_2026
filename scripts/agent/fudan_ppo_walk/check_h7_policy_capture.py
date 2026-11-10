"""Compare real H7 Fudan outputs against the frozen PT using observable history.

Requires a matching successful publication/reset identity in the timing stream,
five consecutive observed frames and no clipping ambiguity. Never opens hardware.
"""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from ct_mjlab.fudan_ppo_walk.export import Policy
from ct_mjlab.fudan_ppo_walk.policy import load_policy

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'sim2sim'))
import real2sim_huananhu as mirror
from h7_latency_capture import LatencyParser


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    meta,_,_=mirror.load_session(args.session)
    parser,latency=mirror.FrameParser(),LatencyParser()
    frames=[];published=set();sdk_us=[];pd_us=[]
    for event in mirror.iter_session_events(args.session):
        if event['type']!='data':continue
        chunk=bytes.fromhex(event['data_hex'])
        frames.extend(parser.feed(chunk))
        for packet in latency.feed(chunk):
            for row in packet['records']:
                if row[1]==2:
                    if row[3]==4:published.add((row[7],row[6]))
                    if row[11]!=0xffffffff:sdk_us.append(row[11])
                elif row[1]==3 and row[3]&1:pd_us.append((row[5]-row[4])/1000)
    network,metadata=load_policy(args.checkpoint)
    assert metadata['contract']=='fudan-ppo-walk-physical25-history125-critic144-v1'
    assert metadata['reset_contract'].endswith('unclipped-history-v2')
    policy=Policy(network).eval()
    window=deque(maxlen=5);last=None;seen=set();skipped=Counter();results=[];inputs=[];history_inputs=[]
    for frame in frames:
        if frame[4]!=3 or frame[152]!=0 or frame[154]!=3 or not int(frame[153])&4 or not int(frame[47])&1:
            continue
        key=mirror.read_u32(frame[156:158]),mirror.read_u32(frame[48:50])
        if key in seen:continue
        seen.add(key)
        if key not in published:
            skipped['no_matching_successful_timing_publication']+=1
            continue
        if last is None or key[0]!=last[0] or (key[1]-last[1])%2**32!=1:
            window.clear()
        last=key
        obs=frame[57:82].astype(np.float32)
        # Clip boundaries cannot reveal the v2 pre-clip FIFO; retain that limitation.
        if not np.isfinite(obs).all() or np.any(np.abs(obs)>=100):
            window.clear();skipped['unobservable_preclip_history']+=1;continue
        window.append(obs)
        if len(window)<5:
            skipped['need_five_consecutive_frames']+=1;continue
        history=np.concatenate(list(window))
        with torch.no_grad():
            expected=policy(torch.from_numpy(obs[None]),torch.from_numpy(history[None])).clamp(-100,100).numpy()[0]
        actual=frame[96:102].astype(np.float32)
        error=float(np.abs(expected-actual).max())
        results.append(dict(reset_count=key[0],policy_sequence=key[1],max_abs_error=error,
                            expected=expected.tolist(),actual=actual.tolist()))
        inputs.append(obs);history_inputs.append(history)
    maximum=max((r['max_abs_error'] for r in results),default=None)
    numerical_pass=bool(len(results)>=20 and maximum is not None and np.isfinite(maximum) and maximum<=1e-4)
    def duration(values):
        return dict(count=len(values),min_us=float(min(values)),max_us=float(max(values)),
                    median_us=float(np.median(values))) if values else dict(count=0)
    report=dict(checked_samples_pass=numerical_pass,atol=1e-4,minimum_samples=20,checked_samples=len(results),
                max_abs_error=maximum,successful_timing_publications=len(published),
                distinct_observed_publications=len(seen),skipped=dict(skipped),sdk_duration=duration(sdk_us),
                pd_duration=duration(pd_us),observed_sdk_within_10ms=bool(sdk_us and max(sdk_us)<10000),
                observed_pd_computation_within_2ms=bool(pd_us and max(pd_us)<2000),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                stream_sha256=meta['stream_sha256'],session=str(args.session),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='actual board published actions versus frozen PT; FIFO reconstructed only from five consecutive nonclipped frames with matching timing/reset IDs',
                limitations=['not direct readback of all 125 board history values',
                             'at least four initial frames per contiguous run are explicitly unmeasured',
                             'sampled worst duration is not a bound for all future operation or full motion acceptance',
                             'PD duration is arithmetic time, not CAN/motor response time'],results=results)
    np.savez_compressed(args.out/'reference-inputs.npz',obs=np.asarray(inputs,dtype=np.float32).reshape(-1,25),
                        history=np.asarray(history_inputs,dtype=np.float32).reshape(-1,125))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    if not numerical_pass:raise SystemExit(2)


if __name__=='__main__':main()
