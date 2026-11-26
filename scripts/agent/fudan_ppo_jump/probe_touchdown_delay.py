"""在同一组平地初态上改变触轮后交接延时，分离第一次跳跃与落地后的再次腾空。"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from . import evaluate as bars
from . import handoff
from . import measure_flat as flat


def run(output, checkpoint):
    walk, jump, saved = flat.policies(checkpoint, 'cuda:0')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(model_sha256=sha(checkpoint), iteration=saved['iterations'], updates=0, delays={})
    for delay in (200, 40, 80):
        folder = output/f'{delay}ms'
        with patch.object(handoff, 'LANDING', dict(handoff.LANDING, buffer_ticks=delay//10)):
            profile = flat.collect_profile(folder, 'walk_jump_walk', walk, jump, landing_handoff=True)
            rows = []
            for item in json.loads((folder/'cases.json').read_text()):
                name = f"start{item['id']:02}.npz"
                with np.load(folder/name) as sample, np.load(output/'200ms'/name) as base:
                    assert all(np.array_equal(base[k],sample[k]) for k in ('qpos0','qvel0','history0'))
                    t, modes = sample['trace'], sample['mode']
                    active = np.flatnonzero(modes)
                    assert len(active)
                    expected, touchdown = handoff.expected_landing_mode(t,int(active[0]))
                    assert np.array_equal(expected,modes)
                    event = next((j for j in item['jumps'] if j['landed']),None)
                    if event is None:
                        rows.append(dict(id=item['id'],complete=False))
                        continue
                    land = round(1000*(event['takeoff_s']+event['airtime_s']))
                    w = t[land:land+1000]
                    assert touchdown>=0 and len(w)==1000
                    switch = 10*int(active[-1]+1)
                    rebounds = [(a,b) for a,b in flat.intervals(~(w[:,8:10]>.5).any(1))
                        if b-a>=30 and w[a:b,6:8].min(1).max()>.02 and w[a:b,19].max()>.15]
                    rows.append(dict(id=item['id'], complete=True,second=bool(rebounds),
                        second_peak_cm=100*max((float(w[a:b,6:8].min(1).max()) for a,b in rebounds),default=0.),
                        contact_ms=int((w[:,10]>1).sum()), vz_at_switch=float(t[switch,19]),
                        actual_switch_ms=switch-land))
        complete = [row for row in rows if row['complete']]
        report['delays'][delay] = dict(cases=len(rows), measured=len(complete),
            stable=profile['first_landing_stable_cases'], second=sum(r['second'] for r in complete),
            median_peak_cm=float(np.median([r['second_peak_cm'] for r in complete])),
            median_contact_ms=float(np.median([r['contact_ms'] for r in complete])),rows=rows)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def test_bars(output, checkpoint, delay):
    params = dict(handoff.LANDING, buffer_ticks=delay//10)
    protocol = dict(bars.LANDING_PROTOCOL,buffer_ticks=delay//10)
    with patch.object(handoff,'LANDING',params), patch.object(bars,'LANDING_PROTOCOL',protocol):
        bars.collect(checkpoint,output,'cuda:0',landing_handoff=True)
        report = bars.assess_recording(output)
    (output/'diagnostic-acceptance.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bar-delay-ms',type=int,choices=(40,80,200))
    a=p.parse_args()
    if a.bar_delay_ms:
        test_bars(a.output,a.checkpoint,a.bar_delay_ms)
    else:
        run(a.output,a.checkpoint)
