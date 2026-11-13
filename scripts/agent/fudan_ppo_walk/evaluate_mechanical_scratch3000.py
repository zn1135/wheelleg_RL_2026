"""Evaluate fresh3000 and previous17679 at identical mechanical wheel PD/physics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.fudan_ppo_walk.run_torque_ablation import run_command
from scripts.agent.fudan_ppo_walk.train_mechanical_3000 import ROOT,PYTHON,LOG


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
    experiment=ROOT/'artifacts/fudan_ppo_walk/20261113-mechanical-scratch3000-01'
    audit=json.loads((experiment/'training-audit.json').read_text());assert audit['passed'] and audit['iterations']==3000
    assert sha(LOG/'model_final.pt')==audit['checkpoint_sha256']
    manifest=json.loads((LOG/'source_sha256.json').read_text())
    for name,digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/') or name.endswith(('.xml','wheel_mechanical.json')):
            assert sha(ROOT/name)==digest,name
    shutil.copy2(__file__,out/'evaluation-launcher.py')
    py=str(PYTHON)
    sources=(('previous17679',ROOT/'artifacts/fudan_ppo_walk/models/20261112_model17679_p60d2_armature2_wheelP045_output045.pt'),
        ('fresh3000',LOG/'model_final.pt'))
    for label,source in sources:
        model=out/'models'/(label+'.pt');shutil.copy2(source,model);assert sha(source)==sha(model)
        parking=out/label
        run_command(out,label+'-parking',[py,'-u','-m','scripts.agent.fudan_ppo_walk.diagnose_parking_posture',
            '--checkpoint',str(model),'--output',str(parking),'--leg-p','60','--leg-d','2',
            '--wheel-p','.1','--wheel-motor-gain','1','--wheel-model','mechanical','--seconds','360'])
        zero=out/(label+'-zero')
        run_command(out,label+'-zero',[py,'-u','-m','scripts.agent.real2sim.replay_mechanical_wheels',
            '--checkpoint',str(model),'--output',str(zero),'--profile','mechanical','--wheel-p','.1'])
        for root,groups in ((parking,('no_external','configured_external')),(zero,('no_external',))):
            for group in groups:
                run_command(out,label+'-'+root.name+'-'+group+'-accept',[str(ROOT/'scripts/mjlab.sh'),'accept',
                    '--task','fudan_ppo_walk','--recording',str(root/group),'--output',str(root/(group+'-acceptance'))])
    report(out)
    write(out/'status.json',dict(status='complete',trajectories=42,training_updates=0,
        full_behavior_accepted=False,comparison=str(out/'comparison.json')))


def report(out):
    rows=[];scores=[];pairs=[]
    for group,suffix in (('no_external',''),('configured_external',''),('zero','-zero')):
        folder_group='no_external' if group=='zero' else group
        folders=[out/(name+suffix)/folder_group for name in ('previous17679','fresh3000')]
        protocols=[json.loads((folder/'protocol.json').read_text()) for folder in folders]
        assert protocols[0]['samples']==protocols[1]['samples']
        assert protocols[0]['physical_wheel_model']==protocols[1]['physical_wheel_model']
        for sample in protocols[0]['samples']:
            sid=sample['id']
            with np.load(folders[0]/'fixtures'/(sid+'_initial.npz')) as x,np.load(folders[1]/'fixtures'/(sid+'_initial.npz')) as y:
                assert x.files==y.files
                for key in x.files:assert np.array_equal(x[key],y[key]),(group,sid,key)
            pairs.append(dict(group=group,id=sid,initial_arrays_identical=True))
        for label,folder,protocol in zip(('previous17679','fresh3000'),folders,protocols):
            score=json.loads((folder.parent/(folder_group+'-acceptance/acceptance.json')).read_text())
            assert all(r['complete'] and r['finite'] for r in score['samples'])
            scores.append(dict(model=label,group=group,passed=score['technical_passed'],total=score['total'],
                passing_ids=[r['sample_id'] for r in score['samples'] if r['passed']]))
            mapping=json.loads((folder/'fixtures/mapping.json').read_text());legs=np.asarray(mapping['motor_q'])[[0,1,3,4]]
            for sample in protocol['samples']:
                sid=sample['id'];end=sample['case']['duration']
                with np.load(folder/'traces'/(sid+'.npz')) as tr:
                    assert np.isfinite(tr['qpos']).all()
                    qm=(tr['qpos_time']>=end-1)&(tr['qpos_time']<end)
                    mask=(tr['time']>=end-5)&(tr['time']<end)
                    w,x,y,z=tr['qpos'][:,3:7].T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                    rows.append(dict(model=label,group=group,sample_id=sid,case=sample['case']['id'],
                        height=sample['case']['segments'][0]['command'][2],
                        leg_last1s_pp_deg=float(np.ptp(np.rad2deg(tr['qpos'][qm][:,legs]),axis=0).max()),
                        pitch_last5s_pp_deg=float(np.ptp(tr['roll_pitch_deg'][mask,1])),
                        pitch_last5s_abs_peak_deg=float(abs(tr['roll_pitch_deg'][mask,1]).max()),
                        speed_last5s_peak_m_s=float(np.linalg.norm(tr['linear_world'][mask,:2],axis=1).max()),
                        yaw_total_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
                        horizontal_displacement_m=float(np.linalg.norm(tr['qpos'][-1,:2]-tr['qpos'][0,:2]))))
    write(out/'comparison.json',dict(rows=rows,scores=scores,pairs=pairs,
        scope='21 paired diagnostic conditions, same mechanical wheels and P60/D2/wheelP0.1; model-native history v2 versus v3; not full987 or real robot',
        training_updates=0,behavior_accepted=False))
    print(json.dumps(dict(scores=scores,zero=[r for r in rows if r['group']=='zero']),indent=2),flush=True)


if __name__=='__main__':main()
