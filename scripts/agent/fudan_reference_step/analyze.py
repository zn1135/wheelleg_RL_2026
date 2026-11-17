"""Offline mechanism measurements from complete per2ms GPU traces."""
import argparse
import json
from pathlib import Path
import numpy as np
from .replay import write


def longest(mask, dt=.002):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    a,b=np.where(edges==1)[0],np.where(edges==-1)[0]
    return float(((b-a)*dt).max(initial=0))


def metrics(path, meta, case):
    with np.load(path) as d:t={k:d[k] for k in d.files}
    h=meta['height'];w=t['wheel_pos'];time=t['time'];n=len(time)
    edge=case.get('edge',0 if meta['kind']=='single' else -10.3)
    direction=case.get('direction',1)
    distance=(w[...,0]-edge)*direction
    relevant=(time>1)&(distance.min(-1)>-.5)&(distance.min(-1)<.8)
    if not relevant.any():relevant=time>1
    before=(time>1)&(distance.max(-1)>-.5)&(distance.max(-1)<-.1)
    vx=t['qvel'][:,0]*direction
    target_height=h if meta['kind']=='single' else .2
    in_x=(distance.min(-1)>.10)&(distance.max(-1)<(2.09-.06 if meta['kind']=='single' else .6-.06))
    width=meta['width'] if meta['kind']=='single' else 8.
    in_y=(np.abs(w[...,1])<width/2-.03).all(-1)
    top=((w[...,2]-(target_height+.06)).__abs__()<.02).all(-1)
    loaded=(t['wheel_support']>1).all(-1)
    achieved=in_x & in_y & top & loaded & (time>1)
    stable=achieved & (np.linalg.norm(t['qvel'][:,:2],axis=-1)<=.05)&(abs(t['pitch_deg'])<=5)&(abs(t['roll_deg'])<=5)&(t['base_force']<=1)&(t['leg_force']<=1)
    air=(t['wheel_force'].__abs__().sum(-1)<1).all(-1)&relevant
    side=(t['wheel_side_force']>1).any(-1)&relevant
    nonwheel=((t['base_force']>1)|(t['leg_force']>1))&relevant
    face=((t['base_force']>1)&(abs(t['base_contact_pos'][:,0]-edge)<.035))|((t['leg_force']>1)&(abs(t['leg_contact_pos'][:,0]-edge)<.035))
    event=np.where((side|achieved|face)&(time>1))[0]
    eventidx=int(event[0]) if len(event) else int(np.argmax(distance.min(-1)))
    before=(time>=time[eventidx]-.20)&(time<=time[eventidx]-.05)
    genuine_air=air&(t['base_force']<=1)&(t['leg_force']<=1)
    wheel_cap=meta.get('wheel_motor_cap',5.)
    torque=t['motor_torque'][relevant];caps=np.array([30,wheel_cap,30,wheel_cap,30,30])
    stand=(time>.5)&(time<1.)
    return dict(id=path.stem,run=path.parent.name,case=case,height_m=meta['height'],width_m=meta['width'],policy_dt=meta['policy_dt'],
        complete=bool(t['complete']),crossed_face=bool(((distance.min(-1)>.1)&(w[...,2].min(-1)>=target_height+.04)&in_y).any()),traversed=longest(achieved)>=.1,
        clean_parked=longest(stable)>=2.,top_support_longest_s=longest(achieved),clean_park_longest_s=longest(stable),
        actual_approach_speed_mean=float(vx[before].mean()) if before.any() else None,
        actual_approach_speed_peak=float(vx[before].max()) if before.any() else None,
        actual_approach_horizontal_speed_mean=float(np.linalg.norm(t['qvel'][before,:2],axis=-1).mean()) if before.any() else None,
        contact_event_t_s=float(time[eventidx]),wheel_relative_x_at_event_m=distance[eventidx].tolist(),
        wheel_z_at_event_m=w[eventidx,:,2].tolist(),wheel_relative_x_max_m=distance.max(0).tolist(),
        wheel_z_max_m=w[relevant,:,2].max(0).tolist(),air_longest_s=longest(air),air_total_s=float(air.sum()*.002),
        whole_robot_air_longest_s=longest(genuine_air),whole_robot_air_total_s=float(genuine_air.sum()*.002),
        speed_at_contact_m_s=float(vx[eventidx]),vertical_speed_at_contact_m_s=float(t['qvel'][eventidx,2]),
        side_contact_total_s=float(side.sum()*.002),nonwheel_contact_total_s=float(nonwheel.sum()*.002),
        base_force_peak_N=float(t['base_force'][relevant].max()),leg_force_peak_N=float(t['leg_force'][relevant].max()),
        base_contact_longest_s=longest((t['base_force']>1)&relevant),leg_contact_longest_s=longest((t['leg_force']>1)&relevant),
        pitch_minmax_deg=[float(t['pitch_deg'][relevant].min()),float(t['pitch_deg'][relevant].max())],
        roll_minmax_deg=[float(t['roll_deg'][relevant].min()),float(t['roll_deg'][relevant].max())],
        max_final_parking_speed_m_s=float(np.linalg.norm(t['qvel'][-1000:,:2],axis=-1).max()),
        virtual_leg_angle_minmax_rad=np.stack([t['virtual_q'][relevant].min(0),t['virtual_q'][relevant].max(0)]).tolist(),
        motor_torque_peak_Nm=np.abs(torque).max(0).tolist(),motor_saturation_fraction=(np.abs(torque)>=caps-.001).mean(0).tolist(),
        end_root_xyz=t['qpos'][-1,:3].tolist(),end_wheel_xyz=w[-1].tolist(),
        standing_root_height_m=float(t['qpos'][stand,2].mean()),
        standing_hip_wheel_distance_m=None,
        observation_note='wheel/contact derived positions lag integrated qpos by one2ms step; flight defined by total force<1N on both wheels; no claim of climbing causality from force timing alone')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[]
    for run in sorted((a.output/'runs').iterdir()):
        if not (run/'execution.json').exists():continue
        meta=json.loads((run/'metadata.json').read_text())
        for i,case in enumerate(meta['specs']):
            f=run/(case.get('id',f'case_{i:03d}')+'.npz')
            if f.exists():rows.append(metrics(f,meta,case))
    dest=a.output/'analysis';dest.mkdir(exist_ok=True)
    write(dest/'metrics.json',rows)
    groups={}
    for r in rows:
        key=r['run'];g=groups.setdefault(key,dict(total=0,traversed=0,clean_parked=0,successful_cases=[]))
        g['total']+=1;g['traversed']+=r['traversed'];g['clean_parked']+=r['clean_parked']
        if r['traversed']:g['successful_cases'].append(r['id'])
    write(dest/'summary.json',groups)
    print(json.dumps(groups,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
