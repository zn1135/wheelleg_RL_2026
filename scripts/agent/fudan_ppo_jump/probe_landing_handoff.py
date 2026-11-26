"""用同一 Jump 模型核对切回行走的时机对平地落地的影响。"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from . import measure_flat as flat

CHECKPOINT = Path('logs/mjlab/fudan_ppo_jump/20261118_resume200_to1200_4096_01/model_final.pt')


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    walk, jump, saved = flat.policies(CHECKPOINT, 'cuda:0')
    assert saved['iterations'] == 1200
    summary = {}
    for ticks in (48, 80, 100):
        folder = output/f'after_{ticks*10}ms'
        with patch.object(flat, 'JUMP_TICKS', ticks):
            data = flat.collect_profile(folder, 'walk_jump_walk', walk, jump)
        contacts = []
        for case in json.loads((folder/'cases.json').read_text()):
            event = next((j for j in case['jumps'] if j['landed']), None)
            if event is None:
                continue
            with np.load(folder/f"start{case['id']:02}.npz") as recorded:
                trace = recorded['trace']
                touch = round(1000*(event['takeoff_s']+event['airtime_s']))
                start = int(np.flatnonzero(recorded['mode'])[0])*10
                contacts.append(dict(world=case['id'], wheel_touch_ms=touch-start,
                    nonwheel_ms=int((trace[touch:touch+1000,10]>1.).sum()),
                    complete_window=len(trace)>=touch+1000))
        data['contacts'] = contacts
        data['median_nonwheel_ms'] = float(np.median([c['nonwheel_ms'] for c in contacts])) if contacts else None
        summary[str(ticks*10)] = data
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for i in range(flat.WORLDS):
        with (np.load(output/'after_480ms'/f'start{i:02}.npz') as a,
              np.load(output/'after_800ms'/f'start{i:02}.npz') as b,
              np.load(output/'after_1000ms'/f'start{i:02}.npz') as c):
            assert all(np.array_equal(a[k],b[k]) and np.array_equal(a[k],c[k])
                       for k in ('qpos0','qvel0','history0'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
