"""按原稳定窗口逐毫秒测量轮载、网格净空、求解器接触对和真实切模时刻。"""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import PhysicalAction
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from . import evaluate, handoff, measure_flat
from .attribute_contacts import SolverPairs


def window(trace, modes, *, physics_dt=.001):
    active=np.flatnonzero(modes)
    substeps=round(.010/physics_dt)
    second=round(1./physics_dt)
    expected,tick=handoff.expected_landing_mode(trace,int(active[0]),8,substeps=substeps)
    assert np.array_equal(expected,modes) and tick>=0
    first=substeps*tick
    wheel=first+int(np.flatnonzero((trace[first:first+substeps,8:10]>.5).any(1))[0])
    both=wheel+int(np.flatnonzero((trace[wheel:wheel+round(.150/physics_dt)+1,8:10]>.5).all(1))[0])
    switch=substeps*(active[-1]+1)
    assert both+second<=len(trace) and switch==substeps*(tick+9)
    return wheel,both,switch


def run(checkpoint, ref, output, *, physics_dt=.001):
    output.mkdir(parents=True,exist_ok=False)
    walk,jump,saved=evaluate.policies(checkpoint,'cuda:0')
    assert saved['iterations']==5200 and sha(checkpoint)==json.loads((ref.parents[1]/'training-audit.json').read_text())['checkpoint_sha256']
    rows=[]
    solver=None
    verts=[]
    meta={}
    second=round(1./physics_dt)
    dt_ms=1 if physics_dt==.001 else physics_dt*1000
    pd_refresh=[]
    original_apply=PhysicalAction.apply_actions

    def probe_pd(action):
        if action._pd_tick==0:
            pd_refresh.append(len(rows))
        original_apply(action)
    frame=JumpEnv.trace_frame
    def capture(env):
        nonlocal solver
        t=frame(env)
        sensor=env.env.scene['jump_wheel']
        if solver is None:
            solver=SolverPairs(env)
            names=list(sensor.primary_names)
            assert names==['lfwheel','rfwheel']
            meta['cols']=[solver.names.index(n) for n in names]
            m=env.env.sim.mj_model
            assert m.opt.timestep==physics_dt
            assert np.isclose(env.dt,.010)
            assert env.action.cfg.pd_decimation*physics_dt==.005
            assert not env.action.cfg.disturbances.action_delay and not env.action.cfg.disturbances.sensor_delay
            meta['geoms']=[g for g,_ in env.wheel_meshes]
            ground=m.geom('terrain')
            assert np.allclose(env.env.sim.data.geom_xpos[:,ground.id,2].cpu(),0)
            for gid,_ in env.wheel_meshes:
                mid=m.geom_dataid[gid];i=m.mesh_vertadr[mid]
                verts.append(np.asarray(m.mesh_vert[i:i+m.mesh_vertnum[mid]],dtype=np.float64))
        s=env.env.sim.data; ids=meta['geoms']
        raw=solver.frame()[:,meta['cols']]
        axes=s.geom_xmat[:,ids].reshape(env.num_envs,2,3,3)[:,:,2,:]
        rows.append(torch.cat((sensor.data.force[:,:,2],raw,s.geom_xpos[:,ids,2],axes.flatten(1)),1))
        return t

    cases=[]
    samples=[]
    with TemporaryDirectory() as tmp, patch.object(handoff,'LANDING',dict(handoff.LANDING,buffer_ticks=8)),patch.object(JumpEnv,'trace_frame',capture),patch.object(PhysicalAction,'apply_actions',probe_pd):
        dest=Path(tmp)/'flat'
        measure_flat.collect_profile(dest,'walk_jump_walk',walk,jump,landing_handoff=True,physics_dt=physics_dt)
        assert pd_refresh==list(range(0,len(rows),round(.005/physics_dt)))
        records=torch.stack(rows).cpu().numpy()
        for i in range(20):
            name=f'start{i:02}.npz'
            with np.load(dest/name) as new,np.load(ref/name) as old:
                assert all(np.array_equal(new[k],old[k]) for k in ('qpos0','qvel0','history0'))
                t=new['trace'];rt=old['trace']
                wheel,both,sw=window(t,new['mode'],physics_dt=physics_dt)
                ow,ob,os=window(rt,old['mode'])
                sl=slice(both,both+second);orig=rt[ob:ob+1000]
                record=records[sl,i,:];force=np.abs(record[:,:2]);pair=record[:,2:4].astype(bool)
                weak=force<=1.; original_weak=orig[:,8:10]<=.5
                assert np.array_equal(~weak,t[sl,8:10]>.5) and not ((~weak)&~pair).any()
                height=t[sl,6:8]*1000
                # 对阈值中断拍用全部碰撞网格顶点、双精度重算轮底，避免依靠微小单精度正值判离地。
                exact=np.full(force.shape,np.nan)
                for side in range(2):
                    mask=weak[:,side]; axis=record[mask,6+side*3:9+side*3].astype(np.float64)
                    exact[mask,side]=(record[mask,4+side].astype(float)+(axis@verts[side].T).min(axis=1))*1000
                confirmed=weak & ~pair & (exact>.0001)   # 高于平面0.1微米且无约束接触
                threshold=weak & pair
                unknown=weak & ~confirmed & ~threshold
                times=(np.arange(both,both+second)-wheel)*dt_ms
                for j,side in zip(*np.where(weak)):
                    samples.append(dict(world=i,side=('L','R')[side],after_wheel_ms=(both+int(j)-wheel)*dt_ms,
                        after_switch_ms=(both+int(j)-sw)*dt_ms,force_N=float(force[j,side]),mesh_gap_mm=float(exact[j,side]),
                        ground_pair=int(pair[j,side]),classification='air' if confirmed[j,side] else 'loaded_below_threshold' if threshold[j,side] else 'near_zero'))
                np.savez_compressed(output/name,ms=times,force_N=force,bottom_mm=height,ground_pair=pair,
                    weak_exact_mm=exact,switch_ms=(sw-wheel)*dt_ms,first_dual_ms=(both-wheel)*dt_ms,dt_ms=dt_ms,
                    original_ms=np.arange(ob,ob+1000)-ow,original_bottom_mm=orig[:,6:8]*1000,
                    original_weak=original_weak,original_switch_ms=os-ow)
                cases.append(dict(world=i,both_ms=(both-wheel)*dt_ms,switch_ms=(sw-wheel)*dt_ms,
                    gap_ms=int(weak.any(1).sum())*dt_ms,left_ms=int(weak[:,0].sum())*dt_ms,right_ms=int(weak[:,1].sum())*dt_ms,
                    both_weak_ms=int(weak.all(1).sum())*dt_ms,original_gap_ms=int(original_weak.any(1).sum()),
                    original_left_ms=int(original_weak[:,0].sum()),original_right_ms=int(original_weak[:,1].sum()),
                    original_both_weak_ms=int(original_weak.all(1).sum()),original_mode_exact=np.array_equal(new['mode'],old['mode']),
                    original_trace_exact=np.array_equal(t,rt),
                    raw_mesh_delta_mm=float(np.nanmax(np.abs(exact-height))) if weak.any() else None))
    for name,data in [('cases',cases),('interruptions',samples)]:
        with (output/(name+'.csv')).open('w') as f:
            fields = data[0].keys() if data else ('world','side','after_wheel_ms','after_switch_ms',
                'force_N','mesh_gap_mm','ground_pair','classification')
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    save_receipt(checkpoint,ref,output)
    timing = dict(dt_ms=dt_ms,pd_ms=5,policy_ms=10,buffer_ms=80,
                  physical_steps=len(rows),pd_refreshes=len(pd_refresh))
    (output/'timing.json').write_text(json.dumps(timing,indent=2)+'\n')


def save_receipt(checkpoint,reference,output):
    with (output/'cases.csv').open() as f:
        cases=list(csv.DictReader(f))
    assert [int(row['world']) for row in cases]==list(range(20))
    expected= json.loads((reference.parents[1]/'training-audit.json').read_text())['checkpoint_sha256']
    assert sha(checkpoint)==expected
    files=[f'start{i:02}.npz' for i in range(20)]
    with np.load(output/files[0]) as d:
        dt_ms=float(d['ms'][1]-d['ms'][0])
    report=dict(checkpoint_sha256=expected,walk_sha256=sha(evaluate.WALK),dt_ms=dt_ms,optimizer_updates=0,
        initial_exact=20,airborne_tolerance_mm=.0001,script_sha256=sha(__file__),
        files_sha256={name:sha(output/name) for name in files},
        reference_sha256={name:sha(reference/name) for name in files})
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','ref','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--physics-step-ms',type=float,choices=(.5,1.),default=1.)
    a=p.parse_args()
    run(a.checkpoint,a.ref,a.output,physics_dt=a.physics_step_ms/1000)
