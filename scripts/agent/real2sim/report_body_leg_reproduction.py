"""Separate saved body pitch and body-relative leg swing; no physics or policy inference."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scripts.agent.real2sim.report_video_leg_reproduction import shape


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--recordings',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);video=[];rows=[];checks=[]
    for window in ('8-10','14-16'):
        folder=args.experiment/f'video-{window}';metadata=json.loads((folder/'measurement.json').read_text())
        with np.load(folder/'all-tracks.npz') as tr:
            t=tr['time'];body=tr['body_image_angle_deg'];leg=tr['leg_relative_body_proxy_deg']
            video.append(dict(window_s=metadata['window_s'],body_image=shape(t,body),leg_relative_body_proxy=shape(t,leg),
                body_angle_tracking_quality={k:metadata['minimum_template_score'][k] for k in ('body_rear','body_front')}))
    for folder in args.recordings:
        protocol=json.loads((folder/'protocol.json').read_text());execution=json.loads((folder/'execution.json').read_text())
        assert execution['complete_traces']==execution['planned']
        model=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));data=mujoco.MjData(model)
        base=model.body('robot/base_link').id;wheel=model.body('robot/rfwheel').id;hip=model.joint('robot/rf0').id
        mapping=json.loads((folder/'fixtures/mapping.json').read_text());qi=np.array(mapping['motor_q'])[[0,1,3,4]]
        initial_error=0.
        references={}
        for sample in protocol['samples']:
            sid=sample['id'];c=sample['candidate'];height=sample['case']['segments'][0]['command'][2];end=sample['case']['duration']
            with np.load(folder/'fixtures'/f'{sid}_initial.npz') as initial:
                if height not in references:references[height]={k:initial[k].copy() for k in ('qpos','qvel','nominal')}
                for k in references[height]:initial_error=max(initial_error,float(abs(initial[k]-references[height][k]).max()))
                assert initial_error<=1e-7 and not np.any(initial['episode_wrench_body'])
                assert int(initial['action_delay_steps'])*2==c.get('action_delay_ms',10)
                assert int(initial['sensor_delay_steps'])*2==c.get('sensor_delay_ms',4)
                legs=np.array(mapping['motor_v'])[[0,1,3,4]]
                assert np.allclose(initial['dof_armature'][legs],.015795*c['armature_scale'],rtol=1e-6,atol=0)
                strength=np.ones(6);strength[[2,5]]=c.get('wheel_strength_scale',1.)
                assert np.allclose(initial['strength'],strength,rtol=1e-6,atol=0)
                assert np.array_equal(initial['kp'],np.ones(6)) and np.array_equal(initial['kd'],np.ones(6))
                assert np.allclose(initial['body_inertia'][base],model.body_inertia[base]*c.get('base_inertia_scale',1.),rtol=1e-6,atol=1e-8)
                assert np.allclose(initial['body_ipos'][base],model.body_ipos[base]+np.array(c.get('base_com_offset_m',[0,0,0])),rtol=1e-6,atol=1e-8)
            with np.load(folder/'traces'/f'{sid}.npz') as tr:
                assert len(tr['qpos'])==round(end*100)+1 and np.isfinite(tr['qpos']).all()
                assert np.isfinite(tr['roll_pitch_deg']).all() and np.isfinite(tr['linear_world']).all()
                for stop in sorted(set((15.,end))):
                    mask=(tr['qpos_time']>=stop-5)&(tr['qpos_time']<stop);t=tr['qpos_time'][mask];q=tr['qpos'][mask]
                    ticks=np.rint(t/.002).astype(int);body=tr['roll_pitch_deg'][ticks,1]
                    relative=[];world=[]
                    for state in q:
                        data.qpos[:]=state;mujoco.mj_kinematics(model,data)
                        v=data.xpos[wheel]-data.xanchor[hip];vb=data.xmat[base].reshape(3,3).T@v
                        relative.append(np.rad2deg(np.arctan2(vb[0],-vb[2])))
                        w,x,y,z=state[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                        world.append(np.rad2deg(np.arctan2(v[0]*np.cos(yaw)+v[1]*np.sin(yaw),-v[2])))
                    joints=np.rad2deg(q[:,qi]);mm=(tr['time']>=stop-5)&(tr['time']<stop)
                    rows.append(dict(recording=str(folder),sample_id=sid,candidate=c,height_m=height,window_s=[stop-5,stop],
                        body_pitch=shape(t,body),body_relative_leg=shape(t,np.asarray(relative)),world_leg=shape(t,np.asarray(world)),
                        body_pitch_first2s_pp_deg=float(np.ptp(body[:200])),body_pitch_last2s_pp_deg=float(np.ptp(body[-200:])),
                        joint_max_pp_deg=float(np.ptp(joints,axis=0).max()),last_second_joint_max_pp_deg=float(np.ptp(joints[-100:],axis=0).max()),
                        body_height_pp_m=float(np.ptp(tr['position'][mm,2])),speed_peak_m_s=float(np.linalg.norm(tr['linear_world'][mm,:2],axis=1).max()),
                        any_tilt_over60=bool((abs(tr['roll_pitch_deg'])>60).any()),
                        trace_sha256=hashlib.sha256((folder/'traces'/f'{sid}.npz').read_bytes()).hexdigest()))
        checks.append(dict(recording=str(folder),complete_finite_traces=execution['complete_traces'],initial_qpos_qvel_nominal_max_difference=initial_error,
            armature_delay_strength_pd_and_body_fixture_checks=True,wall_s=execution['wall_s']))
    result=dict(video=video,rows=rows,checks=checks,physics_steps=0,policy_updates=0,
        limitations='Body image line and body-relative image proxy are uncalibrated2D; GPU body pitch and body-frame hip-wheel angle are3D quantities. Unknown real command height, camera projection and pre-video input history prevent parameter identification.')
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(video=video,checks=checks),indent=2))


if __name__=='__main__':main()
