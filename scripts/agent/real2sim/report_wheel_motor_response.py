"""Audit saved500Hz wheel response against submitted controls and compare parking motion."""
import argparse
import json
from pathlib import Path

import numpy as np
from scripts.agent.real2sim.report_video_leg_reproduction import shape


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--recordings',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[];checks=[]
    for folder in args.recordings:
        protocol=json.loads((folder/'protocol.json').read_text());run=json.loads((folder/'probe-run.json').read_text())
        control=json.loads((folder/'fixtures/actual-control.json').read_text())
        assert (control['leg_p'],control['leg_d'],control['wheel_p'],control['wheel_d'])==(60.,2.,run.get('wheel_p',.1),0.)
        mapping=json.loads((folder/'fixtures/mapping.json').read_text());params=np.array(mapping['wheel_response'],dtype=np.float32)
        summary={r['sample_id']:r for r in json.loads((folder/'parameter-summary.json').read_text())['rows']}
        with np.load(folder/'wheel-response-500hz.npz') as data:
            values=data['values'];time=data['time'];ids=data['sample_ids'].tolist()
        duration=max(s['case']['duration'] for s in protocol['samples'])
        assert values.shape==(round(duration/.002),len(protocol['samples']),2,6) and np.isfinite(values).all()
        worst_delayed=0.;worst_state=0.;worst_control=0.;worst_submit=0.;initial_error=0.;references={}
        for i,sample in enumerate(protocol['samples']):
            sid=sample['id'];assert ids[i]==sid;c=sample['candidate'];height=sample['case']['segments'][0]['command'][2]
            with np.load(folder/'fixtures'/f'{sid}_initial.npz') as f:
                gain=f['wheel_motor_gain'];steps=int(f['wheel_motor_delay_steps']);tau=float(f['wheel_motor_tau_s']);limit=f['wheel_motor_limit_nm']
                assert abs(float(gain)-c.get('wheel_motor_gain',1.))<1e-6
                assert steps==round(c.get('wheel_motor_delay_ms',0.)/2)
                common={k:f[k].copy() for k in ('qpos','qvel','nominal')}
                if height not in references:references[height]=common
                for k in common:initial_error=max(initial_error,float(abs(common[k]-references[height][k]).max()))
                assert initial_error<=1e-7 and not np.any(f['episode_wrench_body'])
                strength=np.ones(6);strength[[2,5]]=c.get('wheel_strength_scale',1.)
                assert np.allclose(f['strength'],strength,rtol=1e-6,atol=0)
                actual_gain=(f['actuator_gainprm'][np.asarray(mapping['motor_ids'])[[2,5]],0]
                    if 'actuator_gainprm' in f else params[:,0])
                if c.get('wheel_mechanical',False):assert np.all(actual_gain==1.) and float(gain)==1. and steps==0 and tau==0.
            req,delayed,state,velocity,response,submitted=[values[:,i,:,j] for j in range(6)]
            assert abs(req).max()<=3.9+1e-5
            expected=np.zeros_like(req)
            if steps:expected[steps:]=req[:-steps]
            else:expected[:]=req
            worst_delayed=max(worst_delayed,float(abs(expected-delayed).max()))
            expected_state=np.zeros_like(state);previous=np.zeros(2,dtype=np.float32)
            alpha=-np.expm1(-np.float32(.002)/np.float32(tau)) if tau else 1.
            for j in range(len(req)):
                previous=expected[j].copy() if not tau else previous+alpha*(expected[j]-previous)
                expected_state[j]=previous
            worst_state=max(worst_state,float(abs(expected_state-state).max()))
            command=expected_state*gain;fraction=np.clip(1-np.maximum(np.sign(command)*velocity,0)/params[:,1],0,1)
            expected_ctrl=command*fraction**params[:,2]
            if c.get('wheel_mechanical',False):expected_ctrl=command
            shaft=expected_ctrl*actual_gain;finite=limit>0
            expected_ctrl[:,finite]=np.clip(shaft[:,finite],-limit[finite],limit[finite])/actual_gain[finite]
            worst_control=max(worst_control,float(abs(expected_ctrl-response).max()))
            worst_submit=max(worst_submit,float(abs(response-submitted).max()))
            base=summary[sid];assert base['complete'] and base['finite']
            with np.load(folder/'traces'/f'{sid}.npz') as tr:
                end=sample['case']['duration'];mask=(tr['time']>=end-5)&(tr['time']<end);body=tr['roll_pitch_deg'][mask,1]
                assert np.isfinite(tr['qpos']).all()
                features=shape(tr['time'][mask],body)
            rows.append(dict(checkpoint_sha256=run['checkpoint_sha256'],sample_id=sid,height_m=height,candidate=c,wheel_p=control['wheel_p'],window_s=[end-5,end],
                last_second_joint_pp_deg=base['last_second_joint_max_pp_deg'],body_pitch=features,
                speed_peak_m_s=base['horizontal_speed_peak_m_s'],any_tilt_over60=base['any_tilt_over60'],
                PD_request_saturation_fraction=float(np.mean(abs(req)>=3.9-1e-5)),
                maximum_submitted_ctrl=float(abs(submitted).max()),maximum_nominal_shaft_torque_nm=float(abs(submitted*actual_gain).max())))
        assert worst_delayed<1e-6 and worst_state<1e-5 and worst_control<1e-5 and worst_submit<1e-6
        checks.append(dict(recording=str(folder),complete_finite_samples=len(protocol['samples']),
            delay_error=worst_delayed,state_error=worst_state,response_formula_error=worst_control,
            actual_ctrl_writeback_error=worst_submit,initial_qpos_qvel_nominal_max_error=initial_error,
            all_samples_same_PD=True,request_limit_nm=3.9))
    result=dict(rows=rows,checks=checks,physics_steps=0,policy_forwards=0,optimizer_updates=0,
        scope='Saved GPU response equations and actual post-scene-write controls; candidates remain unmeasured hypotheses, not real-robot identification')
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
