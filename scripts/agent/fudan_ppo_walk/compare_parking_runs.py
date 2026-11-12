"""Verify paired saved parking fixtures and report leg/body swing plus retained motion passes."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--leg-motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    leg=json.loads(args.leg_motion.read_text())['rows'];groups={};pairing=[]
    runs=[json.loads((folder/'run.json').read_text()) for folder in (args.before,args.after)]
    assert all((r['leg_p'],r['leg_d'])==(60.,2.) for r in runs)
    for group in ('no_external','configured_external'):
        folders=[args.before/group,args.after/group]
        protocols=[json.loads((f/'protocol.json').read_text()) for f in folders]
        assert protocols[0]['samples']==protocols[1]['samples']
        controls=[json.loads((f/'fixtures/actual-control.json').read_text()) for f in folders]
        assert controls[0]==controls[1]
        scores=[json.loads((f.parent/(group+'-acceptance/acceptance.json')).read_text()) for f in folders]
        assert scores[0]['cohort_sha256']==scores[1]['cohort_sha256']
        baseline={s['sample_id']:s for s in scores[0]['samples']};table={}
        for sample in protocols[0]['samples']:
            sid=sample['id'];initials=[]
            for f in folders:
                with np.load(f/'fixtures'/f'{sid}_initial.npz') as d:initials.append({k:d[k].copy() for k in d.files})
            assert initials[0].keys()==initials[1].keys()
            same=all(np.array_equal(initials[0][k],initials[1][k]) for k in initials[0]);assert same
            pairing.append(dict(group=group,sample_id=sid,initial_arrays_exact=True))
        for name,folder,run,score in zip(('before','after'),folders,runs,scores):
            label=Path(run['checkpoint']).stem+f' P{run["leg_p"]:g}/D{run["leg_d"]:g}'
            values=[r for r in leg if r['model']==label and r['group']==group and r['case'].startswith('stop_')]
            assert len(values)==6
            per_case=[]
            for r in values:
                with np.load(folder/'traces'/f'{r["sample_id"]}.npz') as tr:
                    mask=(tr['time']>=14)&(tr['time']<15)
                    assert len(tr['qpos'])==1501 and np.isfinite(tr['qpos']).all()
                    w,x,y,z=tr['qpos'][:,3:7].T
                    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                    per_case.append(dict(sample_id=r['sample_id'],case=r['case'],leg_last1s_pp_deg=r['last_second_max_peak_to_peak_deg'],
                        body_last1s_pitch_pp_deg=float(np.ptp(tr['roll_pitch_deg'][mask,1])),
                        yaw_change_total_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
                        yaw_change_after_stop_deg=float(np.rad2deg(yaw[-1]-yaw[700]))))
            rows=score['samples'];stops=[r for r in rows if r['case']['category'].startswith('stop_')]
            table[name]=dict(total_passed=score['technical_passed'],total=9,stop_passed=sum(s['passed'] for s in stops),stop_total=6,
                median_last_second_joint_pp_deg=float(np.median([r['leg_last1s_pp_deg'] for r in per_case])),
                median_last_second_body_pitch_pp_deg=float(np.median([r['body_last1s_pitch_pp_deg'] for r in per_case])),
                lost_baseline_samples=[r['sample_id'] for r in rows if baseline[r['sample_id']]['passed'] and not r['passed']],
                gained_samples=[r['sample_id'] for r in rows if not baseline[r['sample_id']]['passed'] and r['passed']],samples=per_case)
        groups[group]=table
    result=dict(groups=groups,pairing=pairing,physics_steps=0,policy_forwards=0,optimizer_updates=0,
        scope='Two saved9-case parking groups only; identical control/physics/initial fixtures; not full987 or real-robot acceptance')
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
