"""Check real-time scoring gates at1ms while preserving the historical2ms path."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from ct_mjlab.fudan_ppo_walk.protocol import samples
from scripts.agent.fudan_ppo_walk.check_scoring import fixture
from scripts.agent.fudan_ppo_walk.score_v2 import score
from scripts.agent.motion.acceptance import canonical_sample


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();checks=[]
    def verify(name,value):assert value,name;checks.append(name)
    sample=next(s for s in samples() if s['case']['id']=='zero_0_0_0.26');old=fixture(sample)
    new={k:np.concatenate((np.repeat(v[:-1],2,axis=0),v[-1:]),axis=0) for k,v in old.items()}
    new['time']=np.arange(len(new['time']))*.001
    verify('legacy2ms_pass',canonical_sample(sample,old)['passed'])
    verify('native1ms_pass',canonical_sample(sample,new)['passed'])
    verify('1ms_not_mistaken_for_missing_half_record',score(sample,new,dt=.001)['complete'])
    for duration in (19,20):
        bad=copy.deepcopy(new);bad['wheel_support'][3000:3000+duration]=0.
        result=score(sample,bad,dt=.001)
        verify(f'both_wheels_unloaded_{duration}ms',result['passed']==(duration==19))
        verify(f'reported_run_duration_{duration}ms',result['load_windows'][0]['simultaneous_low_raw']['intervals'][0]['duration_ms']==duration)
    for duration in (20,21):
        bad=copy.deepcopy(new);bad['base_contact'][100:100+duration]=50.
        verify(f'startup_body_contact_{duration}ms',score(sample,bad,dt=.001)['passed']==(duration==20))
    bad=copy.deepcopy(new);bad['linear_world'][3001,0]=.051
    verify('odd1ms_speed_peak_not_discarded',not canonical_sample(sample,bad)['passed'])
    bad=copy.deepcopy(new);bad['time'][3001]+=.0001
    verify('nonuniform1ms_record_rejected',not canonical_sample(sample,bad)['complete'])
    report=dict(passed=True,checks=checks,physics_steps=0,policy_forwards=0,optimizer_updates=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
