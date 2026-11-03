"""Frozen stage-one matrix; sample counts never shrink to fit the time budget."""
import hashlib
import json

PROTOCOL = 'fudan-walk-stage1-v1'


def cases():
    rows = []
    for height in (.22, .26, .30):
        commands = [('zero',0.,0.)]
        commands += [('forward' if v>0 else 'backward',v,0.) for v in (-2.5,-2.,-.25,.25,2.,2.5)]
        commands += [('spin',0.,y) for y in (-2.,2.)]
        commands += [('turn',v,y) for v in (-1.,1.) for y in (-1.,1.)]
        for category,v,y in commands:
            rows.append(dict(id=f'{category}_{v:g}_{y:g}_{height:g}',category=category,
                segments=[dict(start=0.,duration=7.,command=[v,y,height])],duration=7.))
        for v in (-1.,1.):
            rows.append(dict(id=f'stop_{v:g}_{height:g}',category='stop_forward' if v>0 else 'stop_backward',
                segments=[dict(start=0.,duration=7.,command=[v,0.,height]),
                          dict(start=7.,duration=8.,command=[0.,0.,height])],duration=15.))
    for v in (0.,1.):
        rows.append(dict(id=f'height_{v:g}',category='height_standing' if v==0 else 'height_moving',
            segments=[dict(start=7.*i,duration=7.,command=[v,0.,h]) for i,h in enumerate((.22,.30,.26,.22))],
            duration=28.))
    return rows


def samples():
    return [dict(id=f'{c["id"]}__{seed}',case=c,seed=seed,randomized=seed!=50001)
        for c in cases() for seed in [50001,*range(51001,51021)]]


def manifest():
    data=dict(protocol=PROTOCOL,policy_dt=.01,control_dt=.002,samples=samples(),
        train_seed=1,diagnostic_seed=30001,nominal_required=1.,random_required=.95,
        limits=dict(velocity_mae=.20,velocity_band=.30,yaw_mae=.20,yaw_band=.30,
            height_mae=.03,height_band=.05,band_fraction=.95,pitch_deg=10.,
            stop_speed=.10,stop_deadline_s=3.,stop_hold_s=1.,stop_observe_s=5.,fall_angle_deg=60.,fall_base_contact_N=10.))
    data['sha256']=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    return data
