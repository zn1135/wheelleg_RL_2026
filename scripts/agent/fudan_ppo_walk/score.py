"""Stage-one scoring from recorded 500 Hz trajectories; never runs physics."""
import argparse
import json
from pathlib import Path
import numpy as np
from ct_mjlab.fudan_ppo_walk.protocol import manifest,PROTOCOL

DT=.002


def band(values,target,mae,limit):
    errors=np.abs(values-target)
    mean=float(errors.mean()) if errors.size else None
    fraction=float((errors<=limit).mean()) if errors.size else None
    return dict(mae=mean,within_band=fraction,passed=bool(errors.size and mean<=mae and fraction>=.95))


def score(sample,trace,dt=DT):
    case=sample['case'];duration=case['duration'];expected=round(duration/dt)+1
    required=('time','position','linear_body','linear_world','angular_body','roll_pitch_deg',
              'wheel_support','base_contact','weight','terminated','numerical_failure')
    missing=[k for k in required if k not in trace]
    if missing:return dict(sample_id=sample['id'],passed=False,complete=False,missing=missing)
    t=np.asarray(trace['time']);count=len(t)
    complete=(count==expected and np.allclose(t,np.arange(expected)*dt,rtol=0.,atol=1e-7))
    finite=all(np.isfinite(trace[k]).all() for k in required)
    result=dict(sample_id=sample['id'],passed=False,complete=bool(complete),finite=bool(finite),
        observed_samples=count,expected_samples=expected,segments=[])
    result['numerical_failure']=bool(not finite or np.any(trace['numerical_failure']))
    result['terminated']=bool(np.any(trace['terminated']))
    if not complete or not finite:return result
    def window(lo,hi):return (t>=lo-1e-9)&(t<hi-1e-9)
    angles=np.abs(trace['roll_pitch_deg'])
    # 接触或大角度倾覆即失败；重新站起不能覆盖动作过程失败。
    fell=bool((trace['base_contact']>10.).any() or (angles>60.).any())
    integrity=not fell and not np.any(trace['terminated']) and not np.any(trace['numerical_failure'])
    result.update(fell=fell,terminated=bool(np.any(trace['terminated'])),numerical_failure=bool(np.any(trace['numerical_failure'])))
    joint_pass=integrity
    is_stop=case['category'].startswith('stop_')
    for index,segment in enumerate(case['segments']):
        if is_stop and index==1:continue
        lo=segment['start'];hi=lo+segment['duration'];mask=window(lo+2.,hi);response=window(lo,lo+2.)
        vx,yaw,h=segment['command']
        row=dict(command=segment['command'],window=[lo+2.,hi],
            velocity=band(trace['linear_body'][mask,0],vx,.20,.30),
            yaw=band(trace['angular_body'][mask,2],yaw,.20,.30),
            height=band(trace['position'][mask,2],h,.03,.05),
            max_abs_pitch_deg=float(angles[mask,1].max()),response_max_abs_pitch_deg=float(angles[response,1].max()))
        row['passed']=all(row[k]['passed'] for k in ('velocity','yaw','height')) and row['max_abs_pitch_deg']<=10.
        joint_pass &= row['passed'];result['segments'].append(row)
    if is_stop:
        start=7.;speed=np.linalg.norm(trace['linear_world'][:,:2],axis=1)
        stop_tick=round(start/dt);hold=round(1./dt);deadline=round((start+3.)/dt)
        low=speed<=.1
        completed=None
        for end in range(stop_tick+hold,deadline+1):
            if low[end-hold:end].all():completed=end;break
        stop=dict(settle_completed_s=None if completed is None else completed*dt-start,passed=False)
        if completed is not None:
            tail=slice(completed,completed+round(5./dt));anchor=trace['position'][completed,:2]
            stop.update(observation_window=[completed*dt,completed*dt+5.],
                maximum_speed=float(speed[tail].max()),
                maximum_abs_yaw_rate=float(np.abs(trace['angular_body'][tail,2]).max()),
                maximum_drift_m=float(np.linalg.norm(trace['position'][tail,:2]-anchor,axis=-1).max()),
                max_abs_pitch_deg=float(angles[tail,1].max()),
                height=band(trace['position'][tail,2],case['segments'][-1]['command'][2],.03,.05))
            stop['passed']=(stop['maximum_speed']<=.1 and stop['max_abs_pitch_deg']<=10. and stop['height']['passed'])
        result['stop']=stop;joint_pass &= stop['passed']
    last=window(duration-1.,duration);support=trace['wheel_support'][last]
    weight=np.broadcast_to(trace['weight'],(count,))[last]
    conditions=dict(each_wheel=(support>=np.maximum(.1*weight,5.)[:,None]).all(-1),
        total_min=support.sum(-1)>=.8*weight,total_max=support.sum(-1)<=1.2*weight,
        base_contact=trace['base_contact'][last]<=1.,tilt=angles[last].max(-1)<=35.,
        angular_xy=np.linalg.norm(trace['angular_body'][last,:2],axis=-1)<=.5)
    result['balance']={k:bool(value.all()) for k,value in conditions.items()}
    result['last_second_measurements']=dict(wheel_support_min_N=support.min(0).tolist(),
        wheel_support_mean_N=support.mean(0).tolist(),wheel_support_max_N=support.max(0).tolist(),
        total_weight_ratio_min=float((support.sum(-1)/weight).min()),
        total_weight_ratio_max=float((support.sum(-1)/weight).max()),
        base_contact_max_N=float(trace['base_contact'][last].max()),
        tilt_max_deg=float(angles[last].max()),
        angular_xy_max=float(np.linalg.norm(trace['angular_body'][last,:2],axis=-1).max()))
    result['passed']=bool(joint_pass and all(result['balance'].values()))
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();frozen=manifest();rows=[];groups={}
    for sample in frozen['samples']:
        path=a.traces/(sample['id']+'.npz')
        if path.exists():
            with np.load(path) as data:row=score(sample,data)
        else:row=dict(sample_id=sample['id'],passed=False,complete=False,missing=['trajectory'])
        row.update(category=sample['case']['category'],randomized=sample['randomized']);rows.append(row)
        key=f'{row["category"]}:{"random" if sample["randomized"] else "nominal"}'
        group=groups.setdefault(key,dict(total=0,passed=0,complete=0))
        group['total']+=1;group['passed']+=int(row['passed']);group['complete']+=int(row['complete'])
    for key,g in groups.items():
        g['pass_fraction']=g['passed']/g['total'];g['required']=.95 if key.endswith(':random') else 1.
        g['accepted']=g['complete']==g['total'] and g['pass_fraction']>=g['required']
    report=dict(protocol=PROTOCOL,protocol_sha256=frozen['sha256'],total=len(rows),
        technical_thresholds_met=all(g['accepted'] for g in groups.values()),
        final_acceptance='pending_user_video_review',groups=groups,samples=rows)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
