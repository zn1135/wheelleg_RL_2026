"""复算已保存GPU轨迹的失败原因；仅几何回放，零新增积分/网络/学习。"""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from .inspect_critic_scan_stage import ARMS, terrain_path
from .inspect_randomization_4000 import sha, write


def evidence(source):
    manifest=json.loads((source/'evidence-index.json').read_text())['files_sha256']
    checked={}
    def verify(path):
        key=str(path.relative_to(source))
        assert key in manifest and sha(path)==manifest[key],key
        checked[key]=manifest[key]
        return path
    def read(path):return json.loads(verify(path).read_text())
    return verify,read,checked


def source_audit():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run,experiment=ARMS['scan']
    path=run/'source/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk.py'
    manifest=json.loads((run/'source_sha256.json').read_text())
    key='mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk.py'
    assert sha(path)==manifest[key]
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FudanEnv')
    functions={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    raw=next(n.value for n in ast.walk(functions['reward']) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
    terms={k.arg:ast.unparse(k.value) for k in raw.keywords}
    assert terms['collision']=='torch.zeros_like(error)'
    step=ast.unparse(functions['step'])
    assert 'base_contact' not in step and 'undesired_contact' not in step
    events=EventAccumulator(str(run),size_guidance={'scalars':0}).Reload()
    values=events.Scalars('Rewards/collision')
    assert values and all(x.value==0 for x in values)
    config=json.loads((experiment/'budget.json').read_text())['actual_config']
    return dict(passed=True,source=str(path),source_sha256=sha(path),
        collision_raw_expression=terms['collision'],collision_weight=config['rewards']['scales']['collision'],
        all_collision_reward_events_zero=True,collision_reward_events=len(values),
        collision_event_range=[values[0].step,values[-1].step],
        contact_is_not_a_direct_training_termination=True,
        stopping_tracks_forward_velocity_and_yaw_not_horizontal_peak=True,
        velocity_tracking_sigma=config['rewards']['tracking_sigma'],
        reward_at_forward_speed_error_0p1=float(np.exp(-.1**2/config['rewards']['tracking_sigma'])),
        normal_command_high_yaw_zero_vx_probability=.8,
        normal_command_probability=.8,theoretical_reset_high_yaw_only_probability=.64,
        theoretical_not_measured_command_distribution=True)


def geometrical_contact(model,data,trace,tick):
    # 只在已记录姿态上求接触几何；不执行mj_step，不将CPU力当GPU接触力。
    qi=min(int(np.ceil(trace['time'][tick]/.01)),len(trace['qpos'])-1)
    data.qpos[:]=trace['qpos'][qi]
    mujoco.mj_forward(model,data)
    base=model.body('robot/base_link').id
    contacts=[]
    rotation=data.xmat[base].reshape(3,3)
    for c in data.contact:
        ids=[int(i) for i in c.geom]
        if base not in [model.geom_bodyid[i] for i in ids]:continue
        if not any(model.body(model.geom_bodyid[i]).name=='terrain' for i in ids):continue
        contacts.append(dict(geom_ids=ids,geom_names=[model.geom(i).name for i in ids],
            body_names=[model.body(model.geom_bodyid[i]).name for i in ids],
            separation_m=float(c.dist),point_world_m=c.pos.tolist(),
            point_in_base_m=((c.pos-data.xpos[base])@rotation).tolist()))
    return dict(recorded_pose_time_s=float(trace['qpos_time'][qi]),contacts=contacts,
                scope='CPU geometry at saved10ms pose; actual force/time from1ms GPU trace')


def analyze(source,out):
    verify,read,checked=evidence(source)
    result=dict(source_recording=str(source),iteration=5700,physics_steps=0,model_forwards=0,
        optimizer_updates=0,cpu_geometry_only=True,arms={},training_source=source_audit())
    for arm in ARMS:
        terrain_rows=[]
        for folder in ('terrain','terrain-height-extra'):
            root=terrain_path(source,arm,folder,5700)
            rows=read(root/'result.json')['rows']
            model=mujoco.MjModel.from_binary_path(str(verify(root/'fixtures/model.mjb')))
            data=mujoco.MjData(model)
            for r in rows:
                path=verify(root/'traces'/(r['id']+'.npz'))
                with np.load(path) as saved:tr={k:saved[k].copy() for k in saved.files}
                assert all(np.isfinite(v).all() for v in tr.values())
                t=tr['time'];body=np.flatnonzero(tr['base_contact']>10)
                angle=np.flatnonzero(np.abs(tr['roll_pitch_deg']).max(-1)>60)
                row=dict(id=r['id'],family=r['family'],slope_deg=r['slope_deg'],height=r['height'],
                    command_vx=r['command_vx'],complete=r['complete'],reached=r['reached'],
                    passed=r['passed'],drive=r['drive'],stop=r['stop'],
                    endpoint_error_m=r['endpoint_error_m'],mean_drive_vx_m_s=r['mean_drive_vx_m_s'],
                    end_time_s=float(t[-1]),peak_body_contact_N=float(tr['base_contact'].max()),
                    peak_roll_pitch_deg=np.abs(tr['roll_pitch_deg']).max(0).tolist(),
                    first_body_over10_s=float(t[body[0]]) if len(body) else None,
                    first_tilt_over60_s=float(t[angle[0]]) if len(angle) else None)
                if not r['complete']:
                    assert r['failure_reason']=='fall_angle_or_body_contact'
                    assert len(body) or len(angle)
                    first=int(body[0]) if len(body) else int(angle[0])
                    qi=min(int(np.floor(t[first]/.01)),len(tr['qpos'])-1)
                    row.update(cause='body_contact' if len(body) and (not len(angle) or body[0]<=angle[0]) else 'tilt_over60',
                        roll_pitch_at_first_event_deg=tr['roll_pitch_deg'][first].tolist(),
                        before_event_pose_time_s=float(tr['qpos_time'][qi]),
                        before_event_local_height_m=float(tr['height_local_policy'][qi]),
                        before_event_height_deficit_m=float(r['height']-tr['height_local_policy'][qi]),
                        geometry=geometrical_contact(model,data,tr,first))
                else:
                    row['cause']='goal_not_reached' if not r['reached'] else 'reached_but_motion_or_stop_failed'
                    select=(t>=r['duration']-2)&(t<r['duration'])
                    qs=(tr['qpos_time']>=r['duration']-2)&(tr['qpos_time']<r['duration'])
                    q=tr['qpos'][qs,:2];speed=np.linalg.norm(tr['linear_world'][select,:2],axis=-1)
                    row.update(stop_mean_horizontal_m_s=float(speed.mean()),
                        stop_peak_body_vx_vy_m_s=np.abs(tr['linear_body'][select,:2]).max(0).tolist(),
                        stop_net_displacement_m=float(np.linalg.norm(q[-1]-q[0])),
                        stop_total_path_m=float(np.linalg.norm(np.diff(q,axis=0),axis=-1).sum()))
                terrain_rows.append(row)
        nominal=read(source/arm/'nominal/unified-acceptance/acceptance.json')
        failures=[]
        for row in nominal['samples']:
            if row['passed']:continue
            with np.load(verify(source/arm/'nominal/gpu/traces'/(row['sample_id']+'.npz'))) as tr:
                phases=[]
                for phase in row['motion_phases']:
                    if phase['passed']:continue
                    low,high=phase['window'];sel=(tr['time']>=low)&(tr['time']<high)
                    phases.append(dict(**phase,mean_actual_yaw_rad_s=float(tr['angular_body'][sel,2].mean()),
                        mean_actual_vx_m_s=float(tr['linear_body'][sel,0].mean())))
            failures.append(dict(id=row['sample_id'],category=row['case']['category'],
                                 failure_reasons=row['failure_reasons'],phases=phases))
        extras={}
        for label in ('high-yaw','speed3'):
            accepted=read(source/arm/label/'unified-acceptance/acceptance.json')
            bad=[]
            for row in accepted['samples']:
                if row['passed']:continue
                v=dict(id=row['sample_id'],reasons=row['failure_reasons'],phases=row['motion_phases'])
                if 'final_second_attitude' in row['failure_reasons']:
                    with np.load(verify(source/arm/label/'traces'/(row['sample_id']+'.npz'))) as tr:
                        v['existing_attitude_verdict']=row['legacy_v2_diagnostic']['final_second_attitude']
                        # 10ms四元数与同拍角速度配对，剔除沿重力自转分量；不替换原1ms验收。
                        mask=(tr['qpos_time']>=6)&(tr['qpos_time']<7)
                        qw,qx,qy,qz=tr['qpos'][mask,3:7].T
                        gravity=np.stack((2*(qw*qy-qx*qz),-2*(qy*qz+qw*qx),2*(qx*qx+qy*qy)-1),-1)
                        omega=tr['angular_body'][np.rint(tr['qpos_time'][mask]/.001).astype(int)]
                        perpendicular=omega-(omega*gravity).sum(-1)[:,None]*gravity
                        v['gravity_perpendicular_rate_at10ms_peak_rad_s']=float(np.linalg.norm(perpendicular,axis=-1).max())
                        v['diagnostic_only_original_attitude_score_unchanged']=True
                bad.append(v)
            extras[label]=bad
        causes=dict(Counter(r['cause'] for r in terrain_rows))
        result['arms'][arm]=dict(terrain=terrain_rows,terrain_primary_causes=causes,
            nominal_failures=failures,extras_failures=extras)
    scan=result['arms']['scan']['terrain']
    hits=[r for r in scan if r['cause']=='body_contact']
    assert len(hits)==19 and not any(r['first_tilt_over60_s'] is not None for r in hits)
    assert all(r['before_event_height_deficit_m']>0 for r in hits)
    rough=[r for r in scan if r['family']=='rough']
    assert len(rough)==16 and all(r['complete'] and r['reached'] and r['drive']['passed'] and not r['stop']['passed'] for r in rough)
    result['confirmed']=dict(scan_body_contact_failures=19,scan_tilt_over60_recorded=0,
        scan_body_collisions_with_geometrical_confirmation=sum(bool(r['geometry']['contacts']) for r in hits),
        before_contact_height_deficit_minmax_m=[min(r['before_event_height_deficit_m'] for r in hits),max(r['before_event_height_deficit_m'] for r in hits)],
        rough_stop_horizontal_peak_minmax_m_s=[min(r['stop']['metrics']['horizontal_peak'] for r in rough),max(r['stop']['metrics']['horizontal_peak'] for r in rough)],
        rough_stop_mean_horizontal_minmax_m_s=[min(r['stop_mean_horizontal_m_s'] for r in rough),max(r['stop_mean_horizontal_m_s'] for r in rough)],
        rough_stop_net_displacement_minmax_m=[min(r['stop_net_displacement_m'] for r in rough),max(r['stop_net_displacement_m'] for r in rough)])
    result['evidence_sha256']=checked
    write(out/'failure-analysis.json',result)
    shutil.copy2(__file__,out/'analysis-source.py')
    return result


def plot(source,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from scipy.spatial import ConvexHull
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=11)
    root=terrain_path(source,'scan','terrain',5700)
    model=mujoco.MjModel.from_binary_path(str(root/'fixtures/model.mjb'));data=mujoco.MjData(model)
    with np.load(root/'traces/pyramid_3_up_v1.npz') as saved:tr={k:saved[k].copy() for k in saved.files}
    data.qpos[:]=tr['qpos'][-1];mujoco.mj_forward(model,data)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    ax=axes[0,0]
    for body,color,label in [('base_link','#758397','机身'),('lfwheel','#2374ab','车轮')]:
        bid=model.body('robot/'+body).id
        gid=next(i for i in range(model.ngeom) if model.geom_bodyid[i]==bid and model.geom_group[i]==3)
        mesh=model.geom_dataid[gid];start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
        vertices=model.mesh_vert[start:start+count]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
        projected=vertices[:,[0,2]];hull=ConvexHull(projected);polygon=projected[hull.vertices]
        ax.fill(polygon[:,0]-8.,polygon[:,1],color=color,alpha=.75,label=label)
    x=np.linspace(7.8,8.65,100);y=np.maximum(x-8.,0)*np.tan(np.deg2rad(20))
    ax.plot(x-8.,y,color='#926c15',lw=3,label='20°坡面')
    for c in data.contact:
        if model.body('robot/base_link').id in [model.geom_bodyid[int(i)] for i in c.geom]:
            ax.scatter(c.pos[0]-8.,c.pos[2],color='red',s=50,zorder=10)
    ax.set(xlim=(-.18,.65),ylim=(-.02,.5),xlabel='X from ramp foot (m)',ylabel='Z (m)',aspect='equal')
    ax.set_title('0.45秒：机身靠坡侧下沿已碰坡',fontproperties=font);ax.legend(prop=font,fontsize=9)
    ax=axes[0,1]
    ax.plot(tr['qpos_time'],tr['height_local_policy'],label='Actual ground clearance')
    ax.axhline(.26,color='black',ls='--',label='Command0.26m');ax.axvline(.442,color='red',ls=':')
    ax.set(xlabel='Time (s)',ylabel='Clearance (m)');ax.set_title('碰地前离地高度已低于指令',fontproperties=font);ax.legend()
    ax=axes[1,0];other=ax.twinx()
    ax.plot(tr['time'],tr['base_contact'],label='Body contact',color='red')
    ax.axhline(10,color='red',ls='--',alpha=.5)
    other.plot(tr['time'],tr['roll_pitch_deg'][:,1],label='Pitch',color='orange')
    ax.set(xlabel='Time (s)',ylabel='Body contact (N)');other.set_ylabel('Pitch (deg)')
    ax.set_title('触发的是碰撞，机身没有翻到60°',fontproperties=font)
    with np.load(root/'traces/rough_0_dir1_v1.npz') as rough:
        select=(rough['time']>=5)&(rough['time']<7)
        ax=axes[1,1]
        ax.plot(rough['time'][select],np.linalg.norm(rough['linear_world'][select,:2],axis=-1),lw=.9,label='Horizontal speed')
        ax.axhline(.05,color='red',ls='--',label='Parking limit0.05')
        ax.set(xlabel='Time (s)',ylabel='Speed (m/s)');ax.set_title('崎岖路面停车：仍有前后回摆',fontproperties=font);ax.legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'failure-mechanisms.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.recording.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    result=analyze(source,out);plot(source,out)
    print(json.dumps(result['confirmed'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
