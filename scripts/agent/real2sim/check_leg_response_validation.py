"""Check continuous-window input gates against the original physical recording."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay
from scripts.agent.real2sim.validate_leg_response_gpu import motion_blocks


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    session=Path('artifacts/real2sim/h7-leg-capture-20261104/leg-neutral-motion-05')
    decoded=Path('artifacts/real2sim/h7-leg-capture-20261104/leg-neutral-motion-05-output')
    original=replay.read_data(session,decoded)
    extended=replay.read_data(session,decoded,original[-2],23.)
    assert original[-2:]==(315.457,323.457)
    assert extended[-2:]==(315.457,338.457)
    assert np.array_equal(original[6],extended[6][:len(original[6])])
    for duration in (-1.,0.,float('nan'),.0005):
        try:replay.read_data(session,decoded,duration=duration)
        except ValueError:pass
        else:raise AssertionError('Invalid duration was accepted')
    try:replay.read_data(session,decoded,original[-2],10000.)
    except ValueError as error:assert 'cover' in str(error)
    else:raise AssertionError('Truncated recording was accepted')
    blocks,ratios=motion_blocks(original[4],original[5],original[-2],original[-1])
    assert len(blocks)==1 and min(ratios)==0.,'Old unidirectional fitting data cannot satisfy the new protocol'
    times=np.arange(0,12,.01);frames=np.zeros((len(times),160))
    for left,right,rate in ((1,2,-.01),(3,4,.01),(5,6,-.03),(7,8,.03)):
        frames[(times>=left)&(times<right),91]=rate
    blocks,ratios=motion_blocks(frames,times,0,12)
    assert len(blocks)==4 and np.allclose(ratios,3.)
    result=dict(passed=True,scope='recorded input parsing and new bidirectional protocol gates; no hardware or dynamics',
                original_8s_unchanged=True,continuous_23s_prefix_identical=True,
                invalid_duration_rejected=True,truncated_window_rejected=True,
                old_fitting_trajectory_rejected_by_motion_protocol=True,two_speed_reversal_detected=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:json.dump(result,file,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
