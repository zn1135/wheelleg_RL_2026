"""Check saved cross-plant fixtures and report the mechanical wheel replacement limits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/fudan_ppo_walk/20261112-output045-resume14679-3000-01/evaluation/model17679'


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();out=a.experiment
    check=json.loads((out/'implementation-check/report.json').read_text());assert check['passed']
    common=('qpos','qvel','nominal','kp','kd','strength','body_mass','body_inertia','body_ipos','geom_friction','geom_solref',
        'action_delay_steps','sensor_delay_steps','episode_wrench_body','push_seed','push_force_scale')
    pairs=[];scores=[];rows=[]
    groups=[('zero',out/'production-zero-legacy/no_external',out/'production-zero-mechanical/no_external')]
    groups += [(g,BASE/g,out/'mechanical-parking'/g) for g in ('no_external','configured_external')]
    for group,before,after in groups:
        protocols=[json.loads((d/'protocol.json').read_text()) for d in (before,after)]
        assert protocols[0]['validation_backend']==protocols[1]['validation_backend']=='mujoco_warp_gpu'
        new_profile=protocols[1]['physical_wheel_model'];assert new_profile['actuator_gain']==1. and new_profile['extra_output_gain']==1. and not new_profile['speed_shaping']
        assert len(protocols[0]['samples'])==len(protocols[1]['samples'])
        for s,t in zip(protocols[0]['samples'],protocols[1]['samples']):
            assert s['id']==t['id'] and s['case']==t['case'] and s['seed']==t['seed']
            sid=s['id']
            with np.load(before/'fixtures'/f'{sid}_initial.npz') as x,np.load(after/'fixtures'/f'{sid}_initial.npz') as y:
                for field in common:assert np.array_equal(x[field],y[field]),(group,sid,field)
                m=json.loads((after/'fixtures/mapping.json').read_text());wheel=np.asarray(m['motor_v'])[[2,5]];acts=np.asarray(m['motor_ids'])[[2,5]]
                assert np.all(y['actuator_gainprm'][acts,0]==1.) and float(y['wheel_motor_gain'])==1.
                for name in ('armature','damping','frictionloss'):assert np.allclose(y['dof_'+name][wheel],new_profile[name],atol=1e-8,rtol=0)
                assert np.array_equal(np.delete(x['dof_armature'],wheel),np.delete(y['dof_armature'],wheel))
            pairs.append(dict(group=group,id=sid,common_initial_arrays_exact=True,physics_intentionally_different=True))
        for label,folder in (('legacy_curve045',before),('mechanical',after)):
            if group=='zero':score_path=folder.parent/'acceptance/acceptance.json'
            else:score_path=folder.parent/(group+'-acceptance')/'acceptance.json'
            score=json.loads(score_path.read_text());assert all(r['complete'] and r['finite'] for r in score['samples'])
            scores.append(dict(group=group,model=label,passed=score['technical_passed'],total=score['total'],passing_ids=[r['sample_id'] for r in score['samples'] if r['passed']]))
            mapping=json.loads((folder/'fixtures/mapping.json').read_text());legs=np.asarray(mapping['motor_q'])[[0,1,3,4]]
            for s in json.loads((folder/'protocol.json').read_text())['samples']:
                with np.load(folder/'traces'/f'{s["id"]}.npz') as tr:
                    end=s['case']['duration'];qm=(tr['qpos_time']>=end-1)&(tr['qpos_time']<end);mask=(tr['time']>=end-5)&(tr['time']<end)
                    w,x,y,z=tr['qpos'][:,3:7].T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                    rows.append(dict(group=group,model=label,sample_id=s['id'],height=s['case']['segments'][0]['command'][2],
                        leg_last1s_pp_deg=float(np.ptp(np.rad2deg(tr['qpos'][qm][:,legs]),axis=0).max()),
                        pitch_last5s_pp_deg=float(np.ptp(tr['roll_pitch_deg'][mask,1])),
                        horizontal_speed_last5s_peak=float(np.linalg.norm(tr['linear_world'][mask,:2],axis=1).max()),
                        yaw_total_deg=float(np.rad2deg(yaw[-1]-yaw[0]))))
    result=dict(passed=True,scope='Same policy/PD/commands/initial states, intentionally different wheel physics; not hardware calibration or full987',
        pairs=pairs,scores=scores,rows=rows,parameters=check['parameters'],control_check=check['max_force_error_nm'],
        pulse_diagnostic=json.loads((out/'pulse-check/report.json').read_text())['aggregate'],policy_updates=0)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    paths=[ROOT/'mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/wheel_physics.py',ROOT/'resources/robots/chuanliantui_new_1/mjlab/wheel_mechanical.json',
        ROOT/'mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk.py',ROOT/'mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/wheel_motor.py',
        ROOT/'mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk_config.py',ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_action.py',
        ROOT/'mjlab_training/src/ct_mjlab/fudan_ppo_walk/measurement.py',ROOT/'mjlab_training/src/ct_mjlab/rsl_rl/runners/fudan_on_policy_runner.py',Path(__file__)]
    import shutil
    hashes={}
    for path in paths:
        relative=path.relative_to(ROOT);target=out/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(pairs=len(pairs),scores=scores,zero=[r for r in rows if r['group']=='zero']),indent=2))


if __name__=='__main__':main()
