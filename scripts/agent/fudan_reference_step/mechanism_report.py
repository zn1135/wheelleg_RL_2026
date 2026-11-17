"""Offline tables/plots for mechanism.ContactRecorder; no physics integration."""
import json
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .mechanism import load,contact_quantities,CASES,SOURCE
from .replay import write,sha,mujoco_rotation,ONNX_REL
from .analyze import metrics,longest

LOC=['无','地面','立面','上棱角','台面','其他']
SIDES=['左','右']
ARMS=['baseline','torque39','delay15','friction08','policy10','freeze']
LABELS=dict(baseline='原条件',torque39='轮矩3.9Nm',delay15='双延迟15→16ms',friction08='摩擦.8',policy10='策略10ms',freeze='触沿前.3s冻结腿')


def vec(x,digits=1):
    if x is None:return '—'
    return '/'.join(f'{float(v):.{digits}f}' for v in np.ravel(x))


def span(x,digits=1):
    return '—' if np.size(x)==0 else f'{np.min(x):.{digits}f}～{np.max(x):.{digits}f}'


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def save(path,x):write(path,clean(x))


def phase_measurements(out,name,phase):
    p=load(out/'runs/baseline'/(name+'-probe.npz'));t=load(out/'runs/baseline'/(name+'.npz'))
    g=load(out/'geometry-precise'/(name+'.npz'));ts=p['solver_time']
    fn,ft,loaded,loc=contact_quantities(p);n=p['contact_normal'];force=p['contact_force']
    events=sorted(phase['events'],key=lambda x:x['index']);rows=[]
    for k,e in enumerate(events):
        i=e['index'];j=events[k+1]['index'] if k+1<len(events) else min(i+150,len(ts))
        if j<=i:j=min(i+1,len(ts))
        mask=np.zeros(len(ts),bool);mask[i:j]=True
        r=dict(phase=e['phase'],time_s=float(ts[i]),end_s=float(ts[min(j,len(ts)-1)]),index=i,
            virtual_q_rad=g['virtual_q_actual'][i,[0,1,3,4]],
            hip_wheel_body_m=g['hip_wheel_body'][i],
            hip_wheel_length_m=np.linalg.norm(g['hip_wheel_body'][i],axis=-1),
            hip_wheel_angle_from_down_deg=np.rad2deg(np.arctan2(g['hip_wheel_body'][i,:,0],-g['hip_wheel_body'][i,:,2])),
            hip_wheel_left_minus_right_m=g['hip_wheel_body'][i,0]-g['hip_wheel_body'][i,1],
            COM_world_m=g['com_world'][i],root_z_m=p['solver_qpos'][i,2],vertical_speed_m_s=p['solver_qvel'][i,2],
            pitch_deg=np.rad2deg(np.arcsin(np.clip(-mujoco_rotation(p['solver_qpos'][i,3:7])[2,0],-1,1))),
            nonwheel_edge_min_m=g['edge_distance'][mask].min(0),contacts=[],wheel_drive=[],COM_minus_contact_xy_m=[])
        for w in range(2):
            ix=np.where(loaded[i,w])[0]
            if len(ix):
                a=ix[np.argmax(fn[i,w,ix])];r['COM_minus_contact_xy_m'].append(g['com_world'][i,:2]-p['contact_pos'][i,w,a,:2])
            else:r['COM_minus_contact_xy_m'].append(None)
            torque=p['actual_actuator_force'][mask,1+2*w];speed=g['motor_speed'][mask,1+2*w]
            r['wheel_drive'].append(dict(side=SIDES[w],torque_minmax_Nm=[torque.min(),torque.max()],
                saturation_fraction=(abs(torque)>=4.999).mean(),speed_minmax_rad_s=[speed.min(),speed.max()]))
            for l in range(1,6):
                a=loaded[:,w]&(loc[:,w]==l)&mask[:,None]
                if not a.any():continue
                ratio=ft[:,w][a]/(p['contact_mu'][:,w][a]*fn[:,w][a])
                tangent=force[:,w]-fn[:,w,:,None]*n[:,w]
                r['contacts'].append(dict(side=SIDES[w],location=LOC[l],duration_s=a.any(-1).sum()*.002,
                    point_min_m=p['contact_pos'][:,w][a].min(0),point_max_m=p['contact_pos'][:,w][a].max(0),
                    normal_min=n[:,w][a].min(0),normal_max=n[:,w][a].max(0),
                    Fn_minmax_N=[fn[:,w][a].min(),fn[:,w][a].max()],Ft_minmax_N=[ft[:,w][a].min(),ft[:,w][a].max()],
                    ratio_median_p95_max=np.quantile(ratio,[.5,.95,1.]),mu_minmax=[p['contact_mu'][:,w][a].min(),p['contact_mu'][:,w][a].max()],
                    normal_vertical_impulse_Ns=float((fn[:,w]*n[:,w,:,2])[a].sum()*.002),
                    tangent_vertical_impulse_Ns=float(tangent[:,:,2][a].sum()*.002)))
        rows.append(r)
    # Leg-motion onset is a declared threshold relative to 50ms before edge.
    edge=phase['first_edge_s'];start=int(np.argmin(abs(ts-(edge-.05))));onsets=[]
    for w in range(2):
        changed=np.linalg.norm(g['hip_wheel_body'][:,w]-g['hip_wheel_body'][start,w],axis=-1)>.005
        changed&=(ts>=edge-.05)
        ids=np.where(np.convolve(changed.astype(int),np.ones(5,dtype=int),'valid')==5)[0]
        onsets.append(None if len(ids)==0 else float(ts[ids[0]]))
    return rows,dict(leg_motion_5mm_10ms_onset_s=onsets,edge_distance_body_names=g['edge_distance_body_names'].tolist())


def tables(rows,names):
    text=['### 相位时刻：腿与轮驱','',
        '角度按左髋/左虚拟膝/右髋/右虚拟膝排列；髋轮向量为机体系 XYZ；腿向量角从竖直向下算，向机头为正。轮矩和轮速是该相位区间的范围，饱和率按实际±5Nm计算。','',
        '|相位起点→终点(s)|虚拟角(rad)|左/右髋轮向量(mm)|腿长左/右(mm)|腿角左/右(°)|左−右向量(mm)|左轮矩/饱和/轮速|右轮矩/饱和/轮速|',
        '|---|---|---|---|---|---|---|---|']
    for r in rows:
        drives=[f'{span(d["torque_minmax_Nm"],2)}Nm; {100*d["saturation_fraction"]:.1f}%; {span(d["speed_minmax_rad_s"],1)}rad/s' for d in r['wheel_drive']]
        text.append(f'|{r["phase"]} {r["time_s"]:.3f}→{r["end_s"]:.3f}|{vec(r["virtual_q_rad"],3)}|{vec(r["hip_wheel_body_m"][0]*1000)}；{vec(r["hip_wheel_body_m"][1]*1000)}|{vec(r["hip_wheel_length_m"]*1000)}|{vec(r["hip_wheel_angle_from_down_deg"])}|{vec(r["hip_wheel_left_minus_right_m"]*1000)}|{drives[0]}|{drives[1]}|')
    text+=['','### 各轮逐接触位置与力','',
        'Fn为法向力；Ft为去掉法向分量后的切向合力。以下保留每个接触点后按位置分组统计，不把不同方向的接触先合并。点坐标为世界X/Z，法向写X/Z各自范围；全部XYZ、接触几何ID及每2ms样本见probe NPZ。比值为Ft/(μFn)的中位/95分位/最大值，μ实际为1。','',
        '|区间(s)|轮/位置/接触时长|点X/Z范围(m)|法向nx/nz范围|Fn范围(N)|Ft范围(N)|Ft/(μFn)中位/p95/最大|',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        for c in r['contacts']:
            point=f'{c["point_min_m"][0]:.3f}～{c["point_max_m"][0]:.3f} / {c["point_min_m"][2]:.3f}～{c["point_max_m"][2]:.3f}'
            norm=f'{c["normal_min"][0]:.3f}～{c["normal_max"][0]:.3f} / {c["normal_min"][2]:.3f}～{c["normal_max"][2]:.3f}'
            text.append(f'|{r["time_s"]:.3f}～{r["end_s"]:.3f}|{c["side"]}/{c["location"]}/{c["duration_s"]*1000:.0f}ms|{point}|{norm}|{span(c["Fn_minmax_N"])}|{span(c["Ft_minmax_N"])}|{vec(c["ratio_median_p95_max"],2)}|')
    text+=['','MuJoCo源模型采用金字塔摩擦锥（用方形边界近似圆形摩擦上限）；合成Ft/(μFn)可到√2。超过1不能单独认定违反该求解器的摩擦限制。立面判据为nx<−.995、|nz|<.005；台面为nz>.995、|nx|<.005；上棱角还要求接触点距x=0、z=.2各在10/12mm以内。地面要求z<10mm，Fn>1N才计承载。','',
        '### 质心与支撑','',
        '每轮取该时刻Fn最大的真实接触点；无承载点写“—”。COM为全机质量加权质心，表中为COM减接触点的水平XY差；这不是只靠两点投影就判定动态平衡。','',
        '|时刻(s)/相位|COM XYZ(mm)|COM−左接触 XY(mm)|COM−右接触 XY(mm)|pitch(°)|根高(m)|根竖直速度(m/s)|',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        delta=[vec(None if v is None else np.asarray(v)*1000) for v in r['COM_minus_contact_xy_m']]
        text.append(f'|{r["time_s"]:.3f}/{r["phase"]}|{vec(r["COM_world_m"]*1000)}|{delta[0]}|{delta[1]}|{r["pitch_deg"]:.2f}|{r["root_z_m"]:.3f}|{r["vertical_speed_m_s"]:+.3f}|')
    text+=['','### 各部件到上棱角的最小距离','',
        '单位mm，取各相位整个区间最小值；对碰撞凸包求到有限上棱线的最近距离，不是只看网格顶点，也不是离整个平台的间隙。完整2ms曲线见distance图/geometry-precise NPZ。','',
        '|部件|'+'|'.join(f'{r["time_s"]:.3f}～{r["end_s"]:.3f}' for r in rows)+'|',
        '|---|'+'---|'*len(rows)]
    for j,n in enumerate(names):text.append('|'+n+'|'+'|'.join(f'{1000*r["nonwheel_edge_min_m"][j]:.1f}' for r in rows)+'|')
    return '\n'.join(text)+'\n'


def plot_case(out,name,phase):
    p=load(out/'runs/baseline'/(name+'-probe.npz'));t=load(out/'runs/baseline'/(name+'.npz'));g=load(out/'geometry-precise'/(name+'.npz'))
    ts=p['solver_time'];fn,ft,valid,loc=contact_quantities(p)
    end=max(e['time_s'] for e in phase['events'])+.4;mask=(ts>=phase['first_edge_s']-.4)&(ts<=end)
    colors=['#168aad','#c9184a'];fig,axs=plt.subplots(5,2,figsize=(15,15),sharex=True)
    for w in range(2):
        ix=np.argmax(np.where(valid[:,w],fn[:,w],-1),axis=-1);ar=np.arange(len(ts));on=valid[ar,w,ix]
        for l,label in ((1,'ground'),(2,'face'),(3,'corner'),(4,'top')):
            a=valid[:,w]&(loc[:,w]==l)
            axs[0,w].plot(ts,np.where(a,fn[:,w],0).sum(-1),label=label)
        axs[0,w].set_ylabel('Sum Fn by location (N)');axs[0,w].legend()
        axs[1,w].plot(ts,np.where(on,fn[ar,w,ix],np.nan),label='Fn strongest')
        axs[1,w].plot(ts,np.where(on,ft[ar,w,ix],np.nan),label='Ft strongest');axs[1,w].legend();axs[1,w].set_ylabel('N')
        rat=np.where(on,ft[ar,w,ix]/np.maximum(fn[ar,w,ix]*p['contact_mu'][ar,w,ix],1e-9),np.nan)
        axs[2,w].plot(ts,rat);axs[2,w].axhline(1,ls='--',c='k');axs[2,w].axhline(np.sqrt(2),ls=':',c='k');axs[2,w].set_ylabel('Ft / (mu Fn), strongest')
        axs[3,w].plot(ts,np.where(on,p['contact_normal'][ar,w,ix,0],np.nan),label='nx')
        axs[3,w].plot(ts,np.where(on,p['contact_normal'][ar,w,ix,2],np.nan),label='nz');axs[3,w].set_ylabel('normal on wheel');axs[3,w].legend()
        axs[4,w].plot(ts,p['actual_actuator_force'][:,1+2*w],label='actual Nm')
        twin=axs[4,w].twinx();twin.plot(ts,g['motor_speed'][:,1+2*w],c='orange',alpha=.6);twin.set_ylabel('rad/s',color='orange')
        axs[4,w].set_ylabel('Wheel Nm');axs[4,w].set_xlabel('solver time (s)');axs[0,w].set_title(['Left wheel','Right wheel'][w])
    for ax in axs.flat:
        ax.set_xlim(ts[mask][0],ts[mask][-1]);ax.grid(alpha=.25)
        for e in phase['events']:ax.axvline(e['time_s'],c='gray',lw=.7,alpha=.6)
    fig.suptitle(name+' | true solver contacts | mu=1, pyramidal cone');fig.tight_layout();fig.savefig(out/'analysis'/(name+'-contacts.png'),dpi=130);plt.close(fig)
    fig,axs=plt.subplots(6,1,figsize=(12,15),sharex=True)
    for w in range(2):
        axs[0].plot(ts,t['wheel_pos'][:,w,0],color=colors[w],label=f'{"L" if w==0 else "R"} axle x')
        axs[1].plot(ts,t['wheel_pos'][:,w,2],color=colors[w],label=f'{"L" if w==0 else "R"} axle z')
        axs[2].plot(ts,np.linalg.norm(g['hip_wheel_body'][:,w],axis=-1),color=colors[w],label=['L length','R length'][w])
        axs[3].plot(ts,np.rad2deg(g['virtual_q_actual'][:,[0,3][w]]),color=colors[w],label=['L hip','R hip'][w])
        axs[3].plot(ts,np.rad2deg(g['virtual_q_actual'][:,[1,4][w]]),color=colors[w],ls='--',label=['L virtual knee','R virtual knee'][w])
    axs[0].axhline(0,c='k',ls='--');axs[0].plot(ts,g['com_world'][:,0],label='COM x',c='k')
    axs[1].axhline(.26,c='gray',ls='--');axs[1].plot(ts,g['com_world'][:,2],c='k',label='COM z');axs[1].plot(ts,p['solver_qpos'][:,2],ls=':',label='root z')
    pitch=np.array([np.rad2deg(np.arcsin(np.clip(-mujoco_rotation(q)[2,0],-1,1))) for q in p['solver_qpos'][:,3:7]])
    axs[4].plot(ts,pitch,label='pitch deg');axs[4].plot(ts,p['solver_qvel'][:,2]*10,label='root vz x10 m/s')
    names=g['edge_distance_body_names']
    for j,n in enumerate(names):axs[5].plot(ts,g['edge_distance'][:,j]*1000,label=n,lw=2 if j==0 else .8)
    axs[5].legend(ncol=4,fontsize=7);axs[5].set_ylim(0,230)
    for ax,label in zip(axs,['World x (m)','World height (m)','Hip-axle length (m)','Virtual angles (deg)','Pitch / vertical speed','Distance to upper edge (mm)']):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.set_xlim(ts[mask][0],ts[mask][-1])
        if ax!=axs[5]:ax.legend(loc='upper left',ncol=4,fontsize=8)
        for e in phase['events']:ax.axvline(e['time_s'],c='gray',alpha=.5,lw=.7)
    axs[5].set_xlabel('solver time (s)');fig.suptitle(name+' | 2ms kinematics paired to force solver');fig.tight_layout();fig.savefig(out/'analysis'/(name+'-motion.png'),dpi=130);plt.close(fig)


def controls(out):
    rows=[];audit=[]
    for arm in ARMS:
        folder=out/'runs'/arm;meta=json.loads((folder/'metadata.json').read_text())
        execution=json.loads((folder/'execution.json').read_text())
        for case in meta['specs']:
            name=case['id'];path=folder/(name+'.npz');t=load(path);p=load(folder/(name+'-probe.npz'))
            r=metrics(path,meta,case);ts=p['solver_time'];fn,ft,loaded,loc=contact_quantities(p)
            edge=np.where((loaded&np.isin(loc,[2,3,4])).any((1,2))&(ts>1))[0]
            fallen=(abs(t['pitch_deg'])>45)|(abs(t['roll_deg'])>45)|(t['qpos'][:,2]<.12)
            fall=np.where(fallen)[0];firstfall=float(ts[fall[0]]) if len(fall) else None
            event=float(ts[edge[0]]) if len(edge) else None
            valid_approach=event is not None and (firstfall is None or event<firstfall)
            before=(ts>=event-.2)&(ts<=event-.05) if valid_approach else np.zeros(len(ts),bool)
            speed=float(np.linalg.norm(p['solver_qvel'][before,:2],axis=-1).mean()) if before.any() else None
            peakwindow=ts>=1.
            row=dict(arm=arm,id=name,complete=bool(t['complete']),crossed_face=bool(r['crossed_face'] and bool(t['complete'])),
                traversed=bool(r['traversed'] and bool(t['complete'])),top_support_longest_s=r['top_support_longest_s'],
                body_peak_N=float(t['base_force'][peakwindow].max()),leg_peak_N=float(t['leg_force'][peakwindow].max()),
                valid_approach_speed_m_s=speed,first_edge_s=event,first_fall_s=firstfall,
                freeze_requested_s=case.get('freeze_leg_after_s'),policy_dt_s=meta['policy_dt'],
                requested_delay_ms=meta.get('requested_delay_ms',0),effective_delay_ms=meta.get('effective_delay_ms',0),
                wheel_torque_cap_Nm=meta['wheel_motor_cap'],duration_s=len(ts)*.002)
            rows.append(row)
            baseline=load(out/'runs/baseline'/(name+'-probe.npz'))
            finite=all(np.isfinite(a).all() for a in list(t.values())+list(p.values()) if a.dtype.kind in 'fc')
            actual_request_err=float(abs(p['actual_actuator_force']-t['motor_torque']).max())
            cap=meta['wheel_motor_cap'];maxwheel=float(abs(p['actual_actuator_force'][:,[1,3]]).max())
            net_force_error=float(abs(p['contact_force'].sum(2)-t['wheel_force']).max())
            normal_error=float(abs(np.linalg.norm(p['contact_normal'][p['contact_valid']],axis=-1)-1).max())
            a=dict(arm=arm,id=name,steps=len(ts),complete=bool(t['complete']),finite=finite,
                contact_force_sum_vs_legacy_max_error_N=net_force_error,contact_normal_unit_max_error=normal_error,
                actual_actuator_vs_request_max_error_Nm=actual_request_err,wheel_peak_Nm=maxwheel,
                initial_qpos_bitwise_paired=bool(np.array_equal(p['solver_qpos'][0],baseline['solver_qpos'][0])),
                initial_qvel_bitwise_paired=bool(np.array_equal(p['solver_qvel'][0],baseline['solver_qvel'][0])),
                contact_timeline_regular_2ms=bool(np.max(abs(np.diff(ts)-.002))<1e-10),
                negative_Fn_count=int((fn[p['contact_valid']] < -1e-4).sum()))
            if meta.get('effective_delay_ms',0):
                steps=round(meta['effective_delay_ms']/2)
                expected=np.r_[np.zeros_like(t['actions'][:steps]),t['actions'][:-steps]]
                a['action_fixed_delay_max_error']=float(abs(expected-p['applied_actions']).max())
            if arm=='freeze':
                after=ts>=case['freeze_leg_after_s']+meta['policy_dt']
                vals=p['applied_actions'][after][:,[0,1,3,4]]
                a['frozen_leg_action_max_change']=float(abs(vals-vals[0]).max())
            a['wheel_limit_ok']=maxwheel<=cap+1e-5
            assert finite and a['wheel_limit_ok'] and actual_request_err<1e-6 and a['initial_qpos_bitwise_paired'] and a['initial_qvel_bitwise_paired']
            assert net_force_error<1e-6 and normal_error<1e-5 and a['negative_Fn_count']==0
            assert a.get('action_fixed_delay_max_error',0)<1e-6 and a.get('frozen_leg_action_max_change',0)<1e-6
            audit.append(a)
    save(out/'analysis/controls.json',rows);return rows,audit


def additional_evidence(out,phases):
    m=mujoco.MjModel.from_binary_path(str(out/'runs/baseline/model.mjb'));d=mujoco.MjData(m)
    result={}
    for name,phase in phases.items():
        p=load(out/'runs/baseline'/(name+'-probe.npz'));g=load(out/'geometry-precise'/(name+'.npz'))
        t=p['solver_time'];a=phase['events'][1]['index'];end=min(max(e['time_s'] for e in phase['events'])+.3,t[-1])
        mask=(t>=phase['first_edge_s']-.3)&(t<=end);limits=[]
        for j in range(m.njnt):
            if not m.jnt_limited[j]:continue
            q=p['solver_qpos'][mask,int(m.jnt_qposadr[j])];lo,hi=m.jnt_range[j]
            excess=np.maximum(np.maximum(lo-q,q-hi),0)
            limits.append(dict(joint=m.joint(j).name,nominal_range_rad=[lo,hi],actual_range_rad=[q.min(),q.max()],
                excess_max_rad=excess.max(),excess_max_deg=np.rad2deg(excess.max()),excess_duration_s=float((excess>1e-4).sum()*.002)))
        z=[]
        for i in [a]+[None if v is None else int(round(v/.002)) for v in phase['top_times_s']]:
            if i is None:z.append(None);continue
            d.qpos[:]=p['solver_qpos'][i];mujoco.mj_kinematics(m,d)
            z.append(([d.xanchor[m.joint(n).id,2] for n in ('lf0_Joint','rf0_Joint')],
                      [d.body(n).xpos[2] for n in ('l_wheel_Link','r_wheel_Link')]))
        decomposition=[]
        for w in (0,1):
            if z[w+1] is None:continue
            dw=z[w+1][1][w]-z[0][1][w];dh=z[w+1][0][w]-z[0][0][w]
            decomposition.append(dict(side=SIDES[w],wheel_rise_m=dw,hip_rise_m=dh,relative_hip_wheel_vertical_change_m=dw-dh))
        fn,ft,loaded,loc=contact_quantities(p);drive=[]
        for w in (0,1):
            face=(loaded[:,w]&(loc[:,w]==2)).any(-1)&mask
            power=p['actual_actuator_force'][:,1+2*w]*g['motor_speed'][:,1+2*w]
            drive.append(dict(side=SIDES[w],face_wheel_shaft_power_mean_W=power[face].mean(),
                face_wheel_shaft_net_work_J=power[face].sum()*.002,
                face_wheel_torque_saturation_fraction=float((abs(p['actual_actuator_force'][face,1+2*w])>=4.999).mean()),
                wheel_speed_abs_peak_rad_s=abs(g['motor_speed'][mask,1+2*w]).max()))
        result[name]=dict(physical_joint_limits=limits,wheel_rise_kinematic_decomposition=decomposition,wheel_drive=drive)
    save(out/'analysis/additional-evidence.json',result);return result


def report(out):
    ARMS[2]='delay15_zoh';LABELS['delay15_zoh']=LABELS['delay15']
    phases=json.loads((out/'analysis/phases.json').read_text());geometry=json.loads((out/'geometry-precise/audit.json').read_text())
    assert max(v['kinematic_wheel_GPU_max_error_m'] for v in geometry['rows'].values())<2e-7
    rows,audit=controls(out);detail={};onsets={}
    additional=additional_evidence(out,phases)
    for name in CASES:
        rr,extra=phase_measurements(out,name,phases[name]);detail[name]=rr;onsets[name]=extra
        text=f'# {name}：逐相位量测\n\n'+tables(rr,extra['edge_distance_body_names'])
        text+=f'\n![接触](../analysis/{name}-contacts.png)\n\n![运动和距离](../analysis/{name}-motion.png)\n'
        (out/'cases').mkdir(exist_ok=True);(out/'cases'/(name+'.md')).write_text(text)
        plot_case(out,name,phases[name])
    save(out/'analysis/phase-measurements.json',dict(cases=detail,extra=onsets))
    name=CASES[0];p=load(out/'runs/baseline'/(name+'-probe.npz'));t=load(out/'runs/baseline'/(name+'.npz'));g=load(out/'geometry-precise'/(name+'.npz'))
    ts=p['solver_time'];ph=phases[name];a=ph['events'][1]['index'];both=next(e['index'] for e in ph['events'] if e['phase']=='双轮上台')
    fn,ft,loaded,loc=contact_quantities(p);mask=(ts>=ts[a])&(ts<=ts[both])
    m=mujoco.MjModel.from_binary_path(str(out/'runs/baseline/model.mjb'));mass=float(m.body_mass.sum());weight=mass*9.81
    energy=(g['com_world'][both,2]-g['com_world'][a,2])*weight
    base=next(r for r in rows if r['arm']=='baseline' and r['id']==name)
    kinetic=.5*mass*base['valid_approach_speed_m_s']**2
    friction=[]
    for w in range(2):
        face=loaded[:,w]&(loc[:,w]==2)&mask[:,None]
        corner=loaded[:,w]&(loc[:,w]==3)&mask[:,None]
        fz=p['contact_force'][:,w,:,2];normalz=fn[:,w]*p['contact_normal'][:,w,:,2]
        ratio=ft[:,w][face]/(p['contact_mu'][:,w][face]*fn[:,w][face])
        friction.append(dict(side=SIDES[w],face_duration_s=face.any(-1).sum()*.002,
            face_Fn_peak_N=float(fn[:,w][face].max()),face_Ft_peak_N=float(ft[:,w][face].max()),
            face_upward_Ft_peak_N=float((fz-normalz)[face].max()),face_ratio_median_p95=np.quantile(ratio,[.5,.95]),
            face_vertical_normal_impulse_Ns=float(normalz[face].sum()*.002),face_vertical_tangent_impulse_Ns=float((fz-normalz)[face].sum()*.002),
            corner_normal_nz_minmax=[float(p['contact_normal'][:,w,:,2][corner].min()),float(p['contact_normal'][:,w,:,2][corner].max())]))
    save(out/'analysis/mechanism-evidence.json',dict(mass_kg=mass,COM_rise_m=float(g['com_world'][both,2]-g['com_world'][a,2]),
        COM_potential_energy_rise_J=energy,approach_horizontal_kinetic_energy_J=kinetic,contact=friction,
        leg_motion_onsets=onsets[name],events=ph['events']))
    old=json.loads((SOURCE/'analysis/metrics.json').read_text());old=[r for r in old if r['run']=='matched200' and r['id'] in CASES]
    pairing=[dict(id=r['id'],old_traversed=r['traversed'],new_traversed=next(x['traversed'] for x in rows if x['arm']=='baseline' and x['id']==r['id'])) for r in old]
    save(out/'analysis/audit.json',dict(formal_cases=len(rows),complete=sum(r['complete'] for r in rows),
        physics_steps_sum=sum(a['steps'] for a in audit),world_physics_steps=sum(a['steps'] for a in audit),
        optimizer_updates=0,cpu_integration_steps=0,rows=audit,old_60_world_vs_new_4_world_outcomes=pairing,
        geometry_pairing=geometry['rows'],onnx_sha256=sha(out/'source'/ONNX_REL),
        excluded_diagnostics=['delay15 initial action warm-start version (3 complete trajectories; excluded from formal C)',
            'delay14 bracketing attempt aborted on runaway state; no NPZ flushed; log and model preserved; exact15ms not verified']))
    lines=['# 复旦 Upstairs ONNX：200mm 低速倒车登台机制（仅量测）','',
        '结论：该代表例先用轮端对立面的水平压力和向上摩擦爬升，再利用棱角斜向法向支撑，配合左右腿持续、先后的收伸把两轮逐个送上台；既不是单靠轮子滚过3.33倍半径的障碍，也不是主要靠助跑腾空。冻结腿后三例均失败，而轮矩降到3.9Nm三例仍成功。对本机，现有证据中最先缺的是“轮子先接触台阶、低位非轮部件能让开”的几何/动作条件：本机倒车站姿低位机身领先轮前缘约196mm，复旦倒车低位非轮部件反而落后约32mm；本机既有回放也确实是机身先撞沿。双延迟16ms会让这个未经延迟适配的复旦策略在平地启动阶段失稳，但这不能证明已经按延迟训练的本机首先受同一原因限制；精确15ms未验证。','',
        '## 范围与完成情况','',
        '正式完成19/19条、每条10s、每2ms记录：A为3成功+1失败，C为3个预选成功例×5个单因素；总计95000个世界物理步。MuJoCo Warp GPU，冻结原Upstairs ONNX，原2ms物理/PD、4ms策略；原平台200×1148×2090mm。学习更新0，CPU积分0；未启动训练或TensorBoard，未操作分支/提交/推送，未修改fudan_ppo_step、H7和原资产。','',
        '正式C的延迟请求15ms，2ms积分下按保持上一值的离散队列向上取整为16ms；动作队列初值为0、传感队列初值保持复位状态。两类延迟都作用于整段回放（含启动静止），不只在碰沿后启用。另留3条早期队列初值版本及一个14ms补充尝试；后者数值失控时停机、未保存NPZ，均不计正式结果。没有把14/16ms插值写成精确15ms证据。','',
        '## A. 逐相位分解','',
        '|案例（速度指令/接近偏角）|实测接近速度(m/s)|瞬时跨沿/落台|首次触沿(s)|先上台轮|抬轮时刻左/右(s)|首次台面支撑左/右(s)|',
        '|---|---|---|---|---|---|---|']
    for n in CASES:
        r=next(x for x in rows if x['arm']=='baseline' and x['id']==n);ph_=phases[n]
        lines.append(f'|[{n}](cases/{n}.md)|{r["valid_approach_speed_m_s"]:.3f}|{r["crossed_face"]}/{r["traversed"]}|{ph_["first_edge_s"]:.3f}|{ph_["first_side"]}|{vec([np.nan if x is None else x for x in ph_["lift_times_s"]],3)}|{vec([np.nan if x is None else x for x in ph_["top_times_s"]],3)}|')
    lines+=['',
        '“抬轮”定义为轮轴z首次超过75mm；“第一轮上台”为该轮轮心x>0、z>240mm并有台面接触>1N。双轮上台要求同一时刻两轮都满足，不能把两轮分开出现过的上台时刻拼在一起。落台沿用原判据：双轮在台面内、轮轴260±20mm、均承载>1N连续≥.1s。失败例可以有瞬时双轮上台，仍无持续着陆；四例都未连续2s严格停车。','',
        f'代表例右轮在{ph["lift_times_s"][1]:.3f}s先抬，{ph["top_times_s"][1]:.3f}s先上台；左轮在{ph["lift_times_s"][0]:.3f}s已经开始抬，早于右轮完全上台，两个过程有重叠。腿向量相对触沿前50ms基准变化>5mm且持续10ms的起点为左/右{vec(onsets[name]["leg_motion_5mm_10ms_onset_s"],3)}s；“谁先动”只按这一声明阈值判断。','']
    dec=additional[name]['wheel_rise_kinematic_decomposition'][1]
    lim=next(x for x in additional[name]['physical_joint_limits'] if x['joint']=='lf1_Joint')
    lines+=['',
        f"右轮从触沿到首次上台总共升高{1000*dec['wheel_rise_m']:.1f}mm，可按运动学拆为髋部升高{1000*dec['hip_rise_m']:.1f}mm，加上轮轴相对髋部向上移动{1000*dec['relative_hip_wheel_vertical_change_m']:.1f}mm。右腿先略伸长、再从约300mm收至167mm；同时左腿伸至约347mm支撑机身。右轮上台后右腿重新伸长，左腿再收起。该分解是位移账，不是电机做功比例。",'',
        f"源模型左物理膝lf1_Joint在该窗口达到{lim['actual_range_rad'][0]:.4f}rad，名义下限为{lim['nominal_range_rad'][0]:.2f}rad，最大超出{lim['excess_max_rad']:.4f}rad（{lim['excess_max_deg']:.2f}°），持续约{lim['excess_duration_s']*1000:.0f}ms。MuJoCo关节限位采用弹性约束，短暂越限是实际GPU记录，不能把它当作本机硬限位内可直接执行的路径；本轮未改限位。",'',
        '失败例−.5m/s、−20°同样由右轮先上台，但左轮直到9.798s才出现满足定义的台面支撑，没有形成完整落台窗口。它说明轮端先到只是入口条件；本轮没有分离偏角改变受力和策略反馈各自造成的影响。','']
    for f in friction:
        lines.append(f'{f["side"]}轮从首次触沿到双轮上台，在立面累计承载{f["face_duration_s"]*1000:.0f}ms，Fn峰{f["face_Fn_peak_N"]:.1f}N，向上切向力峰{f["face_upward_Ft_peak_N"]:.1f}N，Ft/(μFn)中位/p95为{vec(f["face_ratio_median_p95"],2)}；立面法向竖直冲量{f["face_vertical_normal_impulse_Ns"]:.3f}Ns、切向竖直冲量{f["face_vertical_tangent_impulse_Ns"]:.2f}Ns。')
    lines+=['',f'代表例从触沿到双轮上台，整机COM上升{1000*(energy/weight):.1f}mm，增势能约{energy:.1f}J；接近水平平动能约{kinetic:.2f}J。这里没有计算轮子转动能和气弹簧储能，不能据此精确分账；但结合本次代表例两轮同时离地/全机腾空最长均4ms、立面向上摩擦与腿冻结反证，可排除“主要靠低速惯性抛上去”的解释。','',
        f'![接触、摩擦和实际轮驱](analysis/{name}-contacts.png)','',f'![腿、质心和各杆件距离](analysis/{name}-motion.png)','',
        tables(detail[name],onsets[name]['edge_distance_body_names']),
        '其他两例成功和失败例均有相同结构的相位/接触/力矩/腿/COM/距离表与两张曲线，见上表各案例链接；失败例不是只保存终点判据。','',
        '## B. 两机正向/倒车静态几何四格','',
        '名义站姿按“标称物理、无外扰、实际台阶回放高度指令下的静止姿态”定义，取静止阶段t=.8s保存姿态，随后只做运动学。复旦指令.27m、本机.30m，实测根高各约.259/.303m；没有把虚拟PD零动作参考当作站立关节角，也没有把指令高度直接当实测高度。正向与倒车分别读取对应回放，保留实际偏角。','',
        '前伸量＝沿世界运动方向(+X)，z<200mm的非轮碰撞凸包最前点，减去两轮实际碰撞外形的最前点（圆柱计入轮宽和偏角，网格取实际凸包支撑点）。正数代表非轮部件先到；在z=200mm裁切面加入凸包边与裁切面的交点，避免只取顶点低估前伸量。部件最低高度指整段该部件最低点。','',
        '|机器人/方向|高度指令/实际根高(mm)|前伸量(mm)|限制部件|该部件最低高度(mm)|','|---|---|---|---|---|']
    for r in geometry['static']:
        c=r['limiting_component'];lines.append(f'|{r["robot"]}/{r["body_direction"]}|{r["height_command_m"]*1000:.0f}/{r["root_height_m"]*1000:.1f}|{c["overhang_m"]*1000:+.1f}|{c["body"]}|{c["component_min_z_m"]*1000:.1f}|')
    d=geometry['dynamic'];c=d['limiting_component']
    lines+=['',f'复旦代表例首次触沿t={d["time_s"]:.3f}s的动态同指标：前伸量{c["overhang_m"]*1000:+.1f}mm，限制部件{c["body"]}、最低点{c["component_min_z_m"]*1000:.1f}mm，根高{d["root_height_m"]*1000:.1f}mm。倒车不是把轮子“变大”，而是让轮子能先压到立面，后续腿/机身的路径才有机会继续抬高。','',
        '本机既有200mm倒车例：机身首先碰沿时轮轴距沿255/269mm；这里测得站姿机身领先轮前缘196mm，加上轮半径60mm约256mm，与先碰机身的距离一致。复旦正向低位腿杆领先轮前缘约63mm，而倒车落后约32mm；这个方向差异解释了接触顺序，但单个静止前伸量不能证明整条动态路径无碰撞。代表例机身到上棱角的最小间隙只有约1.7mm，余量也不大。','',
        '[静态四格逐部件数据与定义](geometry-precise/audit.json)；两机资产均未改，CPU积分0。','',
        '## C. 单因素对照','',
        '每项其余条件保持，三例初始qpos/qvel逐位一致；4/3世界小批量运行。非轮峰力取t≥1s全段，包含倒地后接触，故双延迟组大峰力不能当作台沿冲击。速度为世界XY水平合速度，仅在首次有效台阶接触前200～50ms统计；提前失稳不记伪“接近速度”。','',
        '|因素|案例|跨沿|落上台面|机身/腿杆峰力(N)|实际接近速度(m/s)|备注|','|---|---|---|---|---|---|---|']
    for arm in ARMS:
        for r in rows:
            if r['arm']!=arm or r['id']==CASES[-1]:continue
            speed='—' if r['valid_approach_speed_m_s'] is None else f'{r["valid_approach_speed_m_s"]:.3f}'
            note=f'首次失稳{r["first_fall_s"]:.3f}s' if r['first_fall_s'] is not None else ''
            if arm=='freeze':note=f'冻结请求{r["freeze_requested_s"]:.3f}s，下一策略拍生效'
            lines.append(f'|{LABELS[arm]}|{r["id"]}|{r["crossed_face"]}|{r["traversed"]}|{r["body_peak_N"]:.0f}/{r["leg_peak_N"]:.0f}|{speed}|{note}|')
    lines+=['',
        '轮矩3.9Nm仍为3/3落台，故不能把本机失败首先归为轮矩少1.1Nm；摩擦.8和10ms策略各为2/3，最低速例退化。冻结腿为0/3，且冻结前接近速度基本保持，支持持续腿调整是这个闭环机制的必要条件；冻结的是四路网络腿动作，并非锁死关节或腿长。','',
        '双延迟16ms为0/3，失稳阈值定义为|roll|或|pitch|>45°，或根高<.12m。它们在发速度指令前已失稳，属于延迟下的平地闭环不兼容，不能据此量出登台临界延迟，更不能推出本机机构没有能力。精确15ms、延迟适配后的策略、多个因素叠加和随机成功率均未验证。','',
        '## 视频、原始数据与核验','',
        '[正常1倍速、双侧近景接触视频](media-verified/reverse-contact-color-1x.mp4)（左右轮只在渲染中着青/红色；接触点绿/紫，不改碰撞物理）。50fps、500帧、10s，已全量解码；两侧视角都保留整个起步、登台、停车调整。','',
        '- [正式C逐例JSON](analysis/controls.json)、[逐相位完整JSON](analysis/phase-measurements.json)、[机制证据](analysis/mechanism-evidence.json)、[膝限位与位移分解](analysis/additional-evidence.json)、[核验记录](analysis/audit.json)。',
        '- `runs/<因素>/<案例>.npz`：原回放状态/实际提交力矩/接触汇总；`*-probe.npz`：同一求解时刻的qpos/qvel、每轮全部接触点/法向/力/μ/几何ID、实际actuator_force和实际延迟动作。没有只保留最强接触点。',
        '- `geometry-precise/<案例>.npz`：2ms整机COM、机体系髋轮向量、实际虚拟角、实际电机转速、机身与每段腿杆到上棱角距离。',
        '- `solver_time`与接触力对应积分前；旧`time/qpos`是积分后，不能混用。CPU运动学与GPU轮轴位置最大误差<0.0002mm；正式19例有限、实际actuator_force与提交请求差0，逐接触合力与原轮合力一致，单位法向、限幅、冻结/延迟动作逐值检查通过。',
        '- 旧60世界到本次4世界，所选三成功/一失败分类保持，但接触峰力和后续摆动不要求逐点重合。单次固定工况不视为随机成功率，也不是本机/真机或完整运动验收。','',
        '## 复现命令','',
        '以下输出必须换成不存在的新目录。prepare复用已验证固定来源并逐SHA校验；每条GPU命令有自己的deadline。总预算约90分钟，脚本不会自动追加训练。','',
        '```bash',
        'export PYTHONPATH=mjlab_training/src:.',
        'PY=mjlab_training/.venv/bin/python',
        'OUT=artifacts/fudan_reference_step/新的反向机制实验目录',
        '$PY -m scripts.agent.fudan_reference_step.mechanism prepare --output "$OUT"',
        '$PY -m scripts.agent.fudan_reference_step.mechanism run --output "$OUT" --arm baseline --deadline "$(( $(date +%s) + 240 ))"',
        '$PY -m scripts.agent.fudan_reference_step.mechanism phases --output "$OUT"',
        'for ARM in torque39 delay15_zoh friction08 policy10 freeze; do',
        '  $PY -m scripts.agent.fudan_reference_step.mechanism run --output "$OUT" --arm "$ARM" --deadline "$(( $(date +%s) + 240 ))" || break',
        'done',
        '$PY -m scripts.agent.fudan_reference_step.mechanism geometry --output "$OUT"',
        '$PY -m scripts.agent.fudan_reference_step.mechanism report --output "$OUT"',
        'MUJOCO_GL=egl $PY -m scripts.agent.fudan_reference_step.mechanism video --output "$OUT" --deadline "$(( $(date +%s) + 240 ))"',
        '```','',
        '[本轮脚本快照和运行版本](code/manifest.json)。新增`mechanism.py`只用于固定复旦ONNX的单级台阶细粒度接触采集/静态几何/正常速度视频，`mechanism_report.py`离线出表画图；`replay.execute`仅增加可选轮限矩、延迟、细接触探针和失控时保存部分轨迹，原默认调用保持；`analyze.py`按metadata中的轮矩上限计算饱和率。复用了`analyze.metrics`原跨沿/落台判据与`compare`相同的碰撞网格坐标变换，未另建训练/环境体系。','',
        '下一步优先检查本机能否在轮轴距沿约.26m之前，通过提前抬高机身并收腿使低位前伸量降到≤0，再做冻结模型的有限GPU动作回放；有效判据为轮端先接触、全程非轮触沿≤1N、双轮落台支撑≥.1s。此项未在本轮执行，也不授权训练。','']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(out/'README.md'),formal_cases=len(rows),complete=sum(r['complete'] for r in rows)),ensure_ascii=False))
