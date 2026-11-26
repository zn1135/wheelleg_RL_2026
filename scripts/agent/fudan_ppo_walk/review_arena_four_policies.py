"""Review the flat and red-step RMUC integration replays. Behavior is reported separately from routing."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    report = json.loads((path/'report.json').read_text())
    assert report['error'] is None and report['optimizer_updates'] == 0
    with np.load(path/'trace.npz') as data:
        state = data['state']
    roles = np.load(path/'policy-roles.npy')
    contacts = np.load(path/'contact-substeps.npy')
    assert len(roles) == len(state) == report['policy_steps'] and len(contacts) == len(roles)*10
    assert np.isfinite(state).all() and np.isfinite(contacts).all()
    return report, state, roles, contacts


def review(flat, step):
    meta, state, roles, contacts = load(flat)
    jumps = np.flatnonzero(roles == 'jump')
    assert len(jumps) and np.all(np.diff(jumps) == 1)
    assert set(roles) == {'jump','walk'} and roles[-1] == 'walk'
    events = meta['policy_switch']['transitions']
    assert [(e['before'], e['after']) for e in events] == [('walk','jump'),('jump','walk')]
    end = events[-1]
    assert end['reason'] == 'landed' and end['return_tick'] - end['touchdown_tick'] == 9
    assert len(jumps) == end['return_tick'] - end['start_tick']
    first = (jumps[0] + end['touchdown_tick'] - end['start_tick'])*10
    loaded = np.abs(contacts[:, :2]) > 1.
    first += np.flatnonzero(loaded[first:first+10].any(1))[0]
    both = loaded[first:first+1000]
    assert len(both) == 1000
    flat_metrics = dict(jump_steps=len(jumps), return_reason=end['reason'],
        root_rise_m=float(state[jumps,6].max()-state[jumps[0]-1,6]),
        first_second_interrupted_support_ms=int((~both.all(1)).sum()),
        first_second_both_wheels_unloaded_ms=int((~both.any(1)).sum()),
        first_second_nonwheel_contact_ms=int((contacts[first:first+1000,2]>1.).sum()),
        actual_peak_spin_rad_s=float(state[state[:,2]==10.,12].max()))
    _, state, roles, contacts = load(step)
    start = np.r_[0, np.flatnonzero(roles[1:] != roles[:-1])+1]
    assert roles[start].tolist() == ['walk','step','parking','step','walk']
    stop, restart, reset = start[2:]
    assert (state[stop:restart,1:3] == 0.).all() and (state[restart:reset,1]==2.).all()
    speed = np.linalg.norm(state[:,7:9], axis=1)
    approach = contacts[start[1]*10:stop*10]
    step_metrics = dict(red_edge_y_m=-5.405, parking_end_y_m=float(state[restart-1,5]),
        final_parking_second_peak_m_s=float(speed[restart-100:restart].max()),
        final_parking_second_both_wheels_loaded=bool((np.abs(contacts[(restart-100)*10:restart*10,:2])>1.).all()),
        nonwheel_contact_during_approach_ms=int((approach[:,2] > 1.).sum()),
        restart_seconds=(reset-restart)*.01, restart_end_speed_m_s=float(speed[reset-1]))
    return dict(flat=flat_metrics, red_step=step_metrics, routing_passed=True,
                physics='native MuJoCo, one replay per location', formal_acceptance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for arg in ('flat', 'step', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    report = review(args.flat, args.step)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, default=lambda value: value.item())
        stream.write('\n')
    print(report)
