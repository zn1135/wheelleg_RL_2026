"""Independent state-load submission and original-recording response checks; zero integration."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_empirical_joint_load as empirical


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    root=args.experiment;summary=json.loads((root/'report.json').read_text())
    selected=json.loads((root/'selected-empirical-model.json').read_text())
    maximum=0.;results={}
    for name in ('original','wide','slow','fast'):
        directory=root/('final-'+name);report=json.loads((directory/'report.json').read_text())
        model=mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
        mq=[model.joint(n).qposadr[0] for n in empirical.fit.prior.replay.mirror.MOTORS[:4]]
        mv=[model.joint(n).dofadr[0] for n in empirical.fit.prior.replay.mirror.MOTORS[:4]]
        sid=[model.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
        aid=[model.actuator(n+'_motor').id for n in empirical.fit.prior.replay.mirror.MOTORS]
        with np.load(directory/'physics.npz') as saved:
            q=saved['qpos'][:,:,mq];v=saved['qvel'][:,:,mv];applied=saved['applied_force'][:,:,mv];ctrl=saved['ctrl']
        assert np.max(abs(ctrl[:,:,sid]))==0.
        inputs=np.load(Path('artifacts/real2sim/h7-static-constrained-joint-refit-20261105-02/inputs')/name/'replay.npz')
        assert np.max(abs(ctrl[:,0,aid]-np.repeat(inputs['submitted_model_torque'][:,0].astype(np.float32),5,axis=0)))==0.
        offsets=np.array(json.loads((directory/'initial-load-offsets.json').read_text())['offsets_nm'][0])
        for branch in (0,1):
            memory=empirical.memory_path(q[:,branch],selected['delta_rad'])
            travel=np.vstack((np.zeros(4),np.cumsum(abs(np.diff(q[:,branch],axis=0)),axis=0)))
            for side,leg in enumerate(selected['legs']):
                sl=slice(2*side,2*side+2)
                for first in range(0,len(q),1000):
                    last=min(first+1000,len(q));x=np.column_stack((q[first:last,branch,sl],v[first:last,branch,sl],memory[first:last,sl]))
                    phi=empirical.basis((x-leg['mean'])/leg['scale'],np.array(leg['centers']),selected['width'])
                    fade=np.exp(-travel[first:last,sl].sum(1,keepdims=True)/.05)
                    force=-np.clip(phi@np.array(leg['coefficients']).T+offsets[sl]*fade,-40.,40.)
                    maximum=max(maximum,float(abs(force-applied[first:last,branch,sl]).max()))
        with np.load(directory/'trials.npz') as saved:
            states=saved['qpos'];real=saved['real_motor_q']
        metadata=json.loads((Path('artifacts/real2sim/h7-static-constrained-joint-refit-20261105-02/inputs')/name/'report.json').read_text())
        config=Path(json.loads((Path(metadata['session'])/'meta.json').read_text())['machine_config']['path'])
        sign,zero=empirical.fit.prior.replay.mirror.load_mapping(config)
        errors=np.degrees(states[:,:,mq]*sign[:4]+zero-real[:,None,:])
        rows={}
        for branch,key in enumerate(('recorded_torque','recorded_target')):
            rmse=np.sqrt(np.mean(errors[:,branch]**2,axis=0))
            assert np.max(abs(rmse-report['results'][0][key]['rmse_deg']))<1e-6
            rows[key]=dict(joint_rmse_deg=rmse.tolist(),combined_rmse_deg=float(np.sqrt(np.mean(rmse**2))))
        rows['feedback_blocked']=report['results'][0]['feedback_blocked'];results[name]=rows
        print('Verified '+name,flush=True)
    assert maximum<.02,maximum
    for path,digest in summary['protected_sources'].items():assert empirical.fit.audit.digest(Path(path))==digest
    expected=json.loads(Path('artifacts/real2sim/h7-static-calibration-restart-20261105-01/usb/meta.json').read_text())['firmware_manifest']['expected_build']['sources']
    assert all(empirical.fit.audit.digest(Path(p))==h for p,h in expected.items())
    args.output.write_text(json.dumps(dict(submission_formula_verified=True,max_empirical_load_discrepancy_nm=maximum,
        results=results,new_environment_steps=0,unchanged_firmware_source_count=len(expected),
        accepted=False,reason='Large original-recording joint error; some feedback rollouts can lock out',
        shared_model_modified=False),indent=2)+'\n')


if __name__=='__main__':main()
