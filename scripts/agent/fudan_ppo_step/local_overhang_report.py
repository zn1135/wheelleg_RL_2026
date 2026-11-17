"""Offline validation, Pareto plots and Chinese report for local_overhang."""
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .local_overhang import Hulls,MOTORS,CAPS,ROOT,reduction,jacobian,statics
from .diagnose_reverse import write,sha
from ct_mjlab.envs.chuanliantui.model import closed_chain_residual,solve_standing_joint_positions
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force
from scripts.agent.fudan_reference_step.mechanism import overhang


def v(x,d=2):
    return '—' if x is None else '/'.join('—' if a is None else f'{a:.{d}f}' for a in np.ravel(x))


def audit_static(out):
    m=mujoco.MjModel.from_binary_path(str(out/'static/model.mjb'));d=mujoco.MjData(m)
    with np.load(out/'static/search.npz') as a:q=a['qpos'];g=a['tree_gravity_generalized'];T=a['closed_tangent_map']
    hs=Hulls(m);checks=[]
    for i in np.unique(np.linspace(0,len(q)-1,25).astype(int)):
        d.qpos[:]=q[i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        tangent_err=float(abs(jacobian(m,d)@T[i]).max());assert tangent_err<1e-7
        cached=hs.measure(d)[0];exact=[overhang(m,d,hs.wids,s)['limiting_component']['overhang_m'] for s in (1,-1)]
        hull_err=float(abs(np.array(cached)-exact).max());assert hull_err<1e-9
        errors=[];eps=1e-6
        for j in range(6):
            energy=[]
            for sign in (-1,1):
                d.qpos[:]=q[i];d.qpos[7:]+=sign*eps*T[i,6:,j]
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
                energy.append(float((m.body_mass*d.xipos[:,2]).sum()*9.81))
            errors.append(abs((energy[1]-energy[0])/(2*eps)-(g[i]@T[i,:,j])))
        assert max(errors)<2e-5
        checks.append(dict(index=int(i),tangent_error=tangent_err,cached_vs_original_hull_error_m=hull_err,gravity_virtual_work_max_error_Nm=max(errors)))
    selected=json.loads((out/'selected.json').read_text())['rows']
    qp=np.load(out/'selected-poses.npz')['qpos'];equilibria=[]
    for row,qpos in zip(selected,qp):
        d.qpos[:]=qpos;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);tangent=reduction(m,d)
        com,offset,gravity,tg,tn,normal=statics(m,d,tangent)
        force=gravity.copy();jac=np.empty((3,m.nv))
        for name,N in zip(('lfwheel','rfwheel'),normal):
            b=m.body(name).id;point=d.xpos[b]-[0,0,.06];mujoco.mj_jac(m,d,jac,None,point,b);force-=jac.T@np.array([0,0,N])
        for side in ('left','right'):
            j=m.joint(side+'_spring2_joint');force[int(j.dofadr[0])]-=gas_spring_force(np.array(d.qpos[int(j.qposadr[0])]),np.array(0.)).item()
        for name,tau in zip(MOTORS,tn):force[int(m.joint(name).dofadr[0])]-=tau
        J=jacobian(m,d);lam=np.linalg.lstsq(J[:,6:].T,force[6:],rcond=1e-10)[0];res=force-J.T@lam
        item=dict(direction=row['direction'],internal_joint_equilibrium_residual=float(abs(res[6:]).max()),
            required_root_wrench_body_order=res[:6].tolist(),root_moment_norm_Nm=float(np.linalg.norm(res[3:6])),normal_N=normal.tolist())
        assert item['internal_joint_equilibrium_residual']<1e-6
        if row['direction']=='forward':assert abs(res).max()<1e-6
        equilibria.append(item)
    # Exact root-height ceiling: maximum front-chain reach, hip vector rotated
    # within pitch bounds, wheel axis target60mm. This is kinematics only.
    js=solve_standing_joint_positions(m,front_joint_targets={'left':(.06,.14),'right':(-.06,-.14)})
    d.qpos[:]=m.qpos0
    for name,value in js.items():d.qpos[int(m.joint(name).qposadr[0])]=value
    d.qpos[:7]=[0,0,0,1,0,0,0];mujoco.mj_kinematics(m,d)
    vec=d.body('lfwheel').xpos-d.xanchor[m.joint('lf0').id];length=float(np.linalg.norm(vec[[0,2]]))
    hip=d.xanchor[m.joint('lf0').id].copy();pitch=-25.;theta=math.radians(pitch)
    hipz=-math.sin(theta)*hip[0]+math.cos(theta)*hip[2];ceiling=.06+length-hipz
    dofnames=['root_x_N','root_y_N','root_z_N','root_rx_Nm','root_ry_Nm','root_rz_Nm']
    for j in range(1,m.njnt):dofnames.append(m.joint(j).name+('_N' if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else '_Nm'))
    result=dict(passed=True,cpu_integration_steps=0,optimizer_updates=0,samples=checks,selected_static_equilibrium=equilibria,
        root_height_kinematic_ceiling_m=ceiling,front_chain_reach_max_m=length,ceiling_limiter='symmetric left knee upper +.14rad, pitch -25deg',
        generalized_gravity_column_names=dofnames,
        note='Static support approximation: vertical ground loads act at axle minus60mmZ, split by lateral COM; wheels unconstrained in roll. Nonzero COM-X requires the listed external base pitch moment; no claim of stationary balance in reverse.')
    write(out/'static/validation.json',result)
    with (out/'static/selected-all-joint-gravity.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['direction',*dofnames])
        for row,qpos in zip(selected,qp):
            d.qpos[:]=qpos;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);_,_,gr,_,_,_=statics(m,d,reduction(m,d));w.writerow([row['direction'],*gr])
    return result


def duration(mask,ts):
    edge=np.diff(np.r_[False,mask,False].astype(int));starts=np.where(edge==1)[0];ends=np.where(edge==-1)[0]
    return max([float((ts[b] if b<len(ts) else ts[-1]+.002)-ts[a]) for a,b in zip(starts,ends)],default=0.)


def audit_contact_prefix(out,rows):
    m=mujoco.MjModel.from_binary_path(str(out/'dynamic-final/view-model.mjb'));d=mujoco.MjData(m)
    wheels=[m.body(n).id for n in ('lfwheel','rfwheel')];edge=m.geom('query_edge').id
    gids=[g for g in range(m.ngeom) if m.geom_bodyid[g] and m.geom_bodyid[g] not in wheels and (m.geom_contype[g] or m.geom_conaffinity[g])]
    audit=[]
    for r in rows:
        with np.load(out/'dynamic-final'/(r['id']+'-geometry.npz')) as a:g={k:a[k] for k in a.files}
        with np.load(r['trace']) as a:t={k:a[k] for k in a.files}
        stop=r['first_contact_s'] if r['first_contact_s'] is not None else r['valid_duration_s']
        moving=(g['time']>= (r['entered_s'] if r['entered_s'] is not None else np.inf))&(g['time']<stop)
        speed=np.interp(g['time'],t['time'],np.linalg.norm(t['solver_qvel'][:,:2],axis=-1))
        r['negative_overhang_moving_longest_s']=duration(moving&(g['overhang']<=0)&(speed>.1),g['time'])
        r['actual_02_to_05_negative_overhang_longest_s']=duration(moving&(g['overhang']<=0)&(speed>=.2)&(speed<=.5),g['time'])
        pre=g['time']<stop
        r['precontact_closed_chain_residual_max_m']=float(g['closed_chain_residual'][pre].max()) if pre.any() else None
        r['precontact_meets_static_1e7_closure_threshold']=bool(pre.any() and r['precontact_closed_chain_residual_max_m']<1e-7)
        if not r['wheel_first']:continue
        n=round(r['first_wheel_s']/.001);minimum=np.inf;at=None
        for i in range(n):
            d.qpos[:]=t['solver_qpos'][i];d.qpos[:3]-=t['origin']+[1.2,0,0];mujoco.mj_kinematics(m,d)
            distances=[mujoco.mj_geomDistance(m,d,gg,edge,10,None)+.000001 for gg in gids]
            value=min(distances)
            if value<minimum:minimum=value;at=dict(time_s=float(t['time'][i]),body=m.body(m.geom_bodyid[gids[int(np.argmin(distances))]]).name)
        r['precontact_upper_edge_distance_min_m']=float(minimum)
        r['primary_wheel_first_legal_to_contact']=bool(minimum>0)
        audit.append(dict(id=r['id'],physics_sample_dt_s=.001,samples=n,minimum_distance_m=float(minimum),at=at))
    write(out/'dynamic-final/contact-distance-1ms-audit.json',dict(rows=audit,cpu_integration_steps=0,
        scope='All1ms poses strictly before first wheel/first platform contact, all nonwheel collision convex bodies. This is a geometric query, not CPU physics.'))
    path=out/'dynamic-final/metrics.json';payload=json.loads(path.read_text());payload['rows']=rows
    payload['scope']='First contact and hard-joint-limit-safe prefixes only. Dynamic closed-chain residual is measured separately and is not certified by the static1e-7m threshold.'
    write(path,payload)


def report(args):
    out=args.output;result=json.loads((out/'static/search.json').read_text());rows=result['rows'];selected=json.loads((out/'selected.json').read_text())['rows']
    validation=audit_static(out);dyn=json.loads((out/'dynamic-final/metrics.json').read_text())['rows']
    audit_contact_prefix(out,dyn)
    plots=out/'plots';plots.mkdir(exist_ok=True);pareto={}
    for side,name in enumerate(('forward','reverse')):
        a=sorted(rows,key=lambda r:(r['root_height_m'],r['overhang_m'][side]));front=[];best=np.inf
        for r in a:
            if r['overhang_m'][side]<best-1e-8:front.append(r);best=r['overhang_m'][side]
        pareto[name]=front
        fig,ax=plt.subplots(figsize=(10,6));c=ax.scatter([r['overhang_m'][side]*1000 for r in rows],[r['root_height_m']*1000 for r in rows],
            c=[r['pitch_deg'] for r in rows],cmap='coolwarm',vmin=-25,vmax=25,s=9,alpha=.45,rasterized=True)
        ax.plot([r['overhang_m'][side]*1000 for r in front],[r['root_height_m']*1000 for r in front],c='k',lw=1.5,label='Pareto: less overhang / less root height')
        rr=selected[side];ax.scatter([rr['overhang_m'][side]*1000],[rr['root_height_m']*1000],s=140,marker='*',c='limegreen',edgecolor='k',label='GPU reference')
        ax.axvline(0,c='k',ls='--');ax.axvspan(-40,0,color='#76c893',alpha=.14);ax.set_xlim(-35,350);ax.set_ylim(298,387)
        ax.set_xlabel('Nonwheel overhang along motion (mm)');ax.set_ylabel('Root height (mm)');ax.set_title(name+' | symmetric grounded closed chain | hard joint limits')
        fig.colorbar(c,ax=ax,label='Body pitch (deg)');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(plots/(name+'-pareto.png'),dpi=150);plt.close(fig)
    write(out/'static/pareto.json',{k:[{x:r[x] for x in ('index','root_height_m','pitch_deg','overhang_m','limiting_body','COM_minus_wheel_mid_m','motor_load_ratio')} for r in a] for k,a in pareto.items()})
    with (out/'static/feasible-poses.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['index','root_height_m','pitch_deg','hip_rad','knee_rad','overhang_forward_m','overhang_reverse_m','limiter_forward','limiter_reverse','nonwheel_min_z_m','COM_minus_wheel_mid_x_m','gravity_load_ratio_with_spring',*['tau_'+n+'_Nm' for n in MOTORS]])
        for r in rows:w.writerow([r['index'],r['root_height_m'],r['pitch_deg'],r['hip_rad'],r['knee_rad'],*r['overhang_m'],*r['limiting_body'],r['nonwheel_min_z_m'],r['COM_minus_wheel_mid_m'][0],r['motor_load_ratio'],*r['spring_compensated_motor_Nm']])
    # One contact-positive low-command example plus all guidance group curves.
    for group in sorted(set((r['direction'],r['guide']) for r in dyn)):
        fig,axs=plt.subplots(4,1,figsize=(11,11),sharex=True)
        for r in dyn:
            if (r['direction'],r['guide'])!=group:continue
            with np.load(out/'dynamic-final'/(r['id']+'-geometry.npz')) as a:g={k:a[k] for k in a.files}
            label=f'v={r["speed"]:.1f}, yaw={r["angle"]:+.0f}'
            axs[0].plot(g['time'],g['overhang']*1000,label=label);axs[1].plot(g['time'],g['root_z']*1000,label=label)
            axs[3].plot(g['time'],g['edge_distance'].min(-1)*1000,label=label)
            with np.load(r['trace']) as raw:
                line=axs[2].plot(raw['time'][:round(r['valid_duration_s']/.001)],np.linalg.norm(raw['solver_qvel'][:round(r['valid_duration_s']/.001),:2],axis=-1),label=label)[0]
                axs[2].plot(raw['time'][:round(r['valid_duration_s']/.001)],abs(raw['commanded_speed'][:round(r['valid_duration_s']/.001)]),c=line.get_color(),ls='--',lw=.8)
            if r['first_contact_s'] is not None:
                for ax in axs:ax.axvline(r['first_contact_s'],alpha=.25,c='gray',lw=.7)
        axs[0].axhline(0,c='k',ls='--');axs[0].set_ylim(-40,300);axs[0].set_ylabel('Actual overhang (mm)');axs[1].set_ylabel('Root height (mm)')
        axs[2].set_ylabel('Actual speed / dashed command (m/s)');axs[3].set_ylabel('Nonwheel to upper edge (mm)');axs[3].set_ylim(0,500);axs[3].set_xlabel('Solver time (s)')
        for ax in axs:ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=7)
        fig.suptitle('/'.join(group)+' | legal prefix only, vertical lines=first platform contact');fig.tight_layout();fig.savefig(plots/('-'.join(group)+'-dynamic.png'),dpi=140);plt.close(fig)
    with (out/'dynamic-final/contact-order.csv').open('w') as f:
        keys=['id','direction','guide','speed','angle','entered_s','entered_axle_distance_m','first_contact','first_contact_s','first_wheel_s','first_nonwheel_s','primary_wheel_first_legal_to_contact','precontact_upper_edge_distance_min_m','negative_overhang_moving_longest_s','actual_approach_speed_m_s','first_limit_stop_s','limit_excess_at_stop_rad','whole_attempt_hard_limits_legal','stop_reason']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in dyn)
    pp=[r for r in dyn if r['primary_wheel_first_legal_to_contact']];hm=[r for r in dyn if r['guide']=='height_matched']
    summary=dict(total_cases=len(dyn),primary_wheel_first_legal_to_contact=sum(r['primary_wheel_first_legal_to_contact'] for r in dyn),
        entire_trial_legal_and_primary=sum(r['whole_attempt_hard_limits_legal'] and r['primary_wheel_first_legal_to_contact'] for r in dyn),
        height_matched_primary=sum(r['primary_wheel_first_legal_to_contact'] for r in hm),
        actual_approach_range_height_matched_m_s=[min(r['actual_approach_speed_m_s'] for r in hm),max(r['actual_approach_speed_m_s'] for r in hm)],
        static_physics_steps=0,gpu_batch_physics_ticks=sum(json.loads((p/'protocol.json').read_text())['physics_ticks'] for p in (out/'gpu').iterdir()),
        gpu_world_steps_including_protective_resets=sum(json.loads((p/'protocol.json').read_text())['physics_ticks']*6 for p in (out/'gpu').iterdir()),
        cpu_integration_steps=0,optimizer_updates=0,
        statement='Geometry and legal-to-first-touch evidence exist forward; full low-speed hard-limit-safe attempt not demonstrated. Reverse static geometry exists but chosen guides did not capture it.')
    write(out/'summary.json',summary)
    lines=['# 本机能否让轮子先碰200mm台沿：静态搜索与有限GPU回放','',
        '结论：本机存在硬关节限位内、闭链闭合、前伸量≤0的对称双轮着地姿态，正向和倒车都能在几何上摆出；正向还可让质心对齐轮轴中点。行驶中，正向高度匹配组6例在合法前缀内均做到轮先碰台阶，负前伸最长连续约1.7s；但实际接近速度为.61～.76m/s，高于.2/.3/.5m/s指令，触沿后也都触限停机。因此已证明“正向入口条件短时可实现”，尚未证明“按要求的实际低速、整段无越限地保持并爬升”。倒车几何解有大质心偏移，两种脚本都未进入目标姿态；这不是倒车所有控制器不可行的证明。','',
        '## 范围与判据','',
        f'静态搜索{result["tried"]}个前链/俯仰组合，得到{len(rows)}个根高≥.30m、无非轮落地、硬限位合法且闭链合格姿态。左右对称；左膝[-.97,+.14]rad与右膝镜像限位的交集；髋部扫描一整圈（3°步长），膝.02rad步长，pitch−25～+25°（2.5°步长）。轮轴可前后摆动，未锁定在根坐标正下方。','',
        '闭链残差<1e−7m、轮轴着地目标误差<.5mm，沿用keyframe_probe的较严阈值；硬限位只容许1e−9rad浮点误差，不用柔性越限扩大可达范围。高度由双轮轴z=.06m的着地条件求出，不以高度指令代替实际根高。双方向都有对称解，按要求没有补查非对称。','',
        f'解析前链伸展及髋偏置给出的根高上限为{validation["root_height_kinematic_ceiling_m"]*1000:.2f}mm，边界是左膝+.14rad及pitch−25°；网格最高{max(r["root_height_m"] for r in rows)*1000:.2f}mm。该上限不是稳定行驶高度。','',
        '前伸量完全沿用上轮mechanism.overhang：碰撞凸包在z=200mm下方的最前点减实际轮前缘；加入凸包边与裁切面的交点。此处只缓存不变的凸包顶点/边，已与原函数逐点比对。小于0代表轮端领先；显示正负号，不把“无接触”当成负前伸。','',
        '## 1. 静态边界与可行姿态','',
        '![正向帕累托](plots/forward-pareto.png)','', '![倒车帕累托](plots/reverse-pareto.png)','',
        '图上每一点都满足静态几何约束，颜色为pitch；黑线为同时减少前伸量和根高的帕累托边界。绿色星号是GPU选用的参考，不以最小根高牺牲质心对齐。','',
        '|方向|负前伸网格姿态数|网格最小前伸(mm)|对应根高(mm)/pitch(°)|限制部件|','|---|---|---|---|---|']
    for i,n in enumerate(('正向','倒车')):
        good=[r for r in rows if r['overhang_m'][i]<=0];best=min(rows,key=lambda r:r['overhang_m'][i]);lines.append(f'|{n}|{len(good)}|{best["overhang_m"][i]*1000:.2f}|{best["root_height_m"]*1000:.1f}/{best["pitch_deg"]:.1f}|{best["limiting_body"][i]}|')
    lines+=['',
        '|选用参考|根高(mm)|pitch(°)|前伸(mm)|非轮最低点(mm)|COM−轮轴中点X/Y(mm)|左膝(rad)|重力＋气弹簧腿矩峰值/40Nm|所需外部根俯仰力矩(Nm)|','|---|---|---|---|---|---|---|---|---|']
    for i,r in enumerate(selected):
        lines.append(f'|{r["direction"]}|{r["root_height_m"]*1000:.1f}|{r["pitch_deg"]:.1f}|{r["overhang_m"][i]*1000:.2f}|{r["nonwheel_min_z_m"]*1000:.1f}|{v(np.array(r["COM_minus_wheel_mid_m"][:2])*1000,2)}|{r["knee_rad"]:.4f}|{max(abs(a) for j,a in enumerate(r["spring_compensated_motor_Nm"]) if j not in (2,5)):.2f}/40|{r["base_holding_pitch_moment_Nm"]:.2f}|')
    lines+=['',
        '正向选用根高360mm、pitch0°、COM水平前后偏移≈0的解，前伸−22.53mm；倒车选用5mm轮端领先余量、左膝距上限20mrad的解，根高327mm、pitch25°，COM前后偏移仍124.5mm，静止时需约16.36Nm外部根俯仰支撑。倒车仅“几何可摆出”，该姿态不等于能仅靠静止双轮平衡。网格内倒车负前伸解最小COM前后偏移约115.4mm；这是网格结果，不是所有连续/非对称姿态的数学最小值。','',
        '### 静态重力力矩的含义','',
        '使用原质量/质心和真实闭链雅可比，通过虚功（关节微动引起的重力做功）换算到6个主动电机。轮端承担整机重力，左右载荷按COM横向位置分配；分别给出无气弹簧和计入名义气弹簧的电机矩。没有把树模型里某一关节的qfrc_bias直接当成闭链电机矩。','',
        '|参考|无气弹簧重力电机矩(Nm)|计入气弹簧后(Nm)|限幅(Nm)|','|---|---|---|---|']
    for r in selected:lines.append(f'|{r["direction"]}|{v(r["loaded_gravity_motor_Nm"],3)}|{v(r["spring_compensated_motor_Nm"],3)}|40/40/3.9/40/40/3.9|')
    lines+=['',
        '顺序为lf0/lf00/lfwheel/rf0/rf00/rfwheel。轮矩很小只说明静止重力载荷，不代表平衡/加速/碰沿时轮矩不重要。正向参考在点支撑近似下通过整机静力平衡方程；倒车还需表中根俯仰力矩，不能只凭腿矩<40Nm称它静态可保持。','',
        '[逐姿态可行表CSV](static/feasible-poses.csv)、[全部关节原始重力CSV](static/selected-all-joint-gravity.csv)、[搜索JSON](static/search.json)、[9343姿态/各关节重力/闭链映射NPZ](static/search.npz)、[连续精化参考](selected.json)、[帕累托表](static/pareto.json)、[正式核验](static/validation.json)。','',
        '## 2. 冻结3000有限GPU回放','',
        '30例，5个小批次×6世界，每例≤12s；固定速度指令.2/.3/.5m/s与偏角0/−10°。原3000 SHA256=dfb221c09648d278d5c5e5c7b78e4d729b7a7a5a56458762b7313dc50831c4f8；1ms GPU积分、5ms原PD、10ms策略，双延迟各精确15ms，摩擦.8，名义物理，单级平台200×1148×2090mm、无侧障碍。未改环境实现、控制增益/限矩、机器人资产或H7，未训练或启动TensorBoard。','',
        '均从原站姿动态进入参考，不强设运行中的根或关节状态。1～3.5s平滑引导，进入条件要求实际前伸≤−2mm连续100ms、根高误差≤25mm、pitch误差≤5°、双轮承载，并在最近轮轴距沿≥.4m处完成；达不到则5s结束，速度指令不释放。进入后再给所列低速指令。','',
        '三种脚本：additive保留四腿NN反馈，加“目标姿态−.9s实际腿姿”的偏置；held将四腿目标渐变为静态姿态＋重力/PD补偿，轮子继续用冻结NN；height_matched仅正向，把高度指令从.30渐变到.35m，同时加相对约.35m静态参考的小腿偏置。第三组同时改变高度指令和偏置基准，不是高度单因素因果试验。','',
        '|策略/方向|进入负前伸参考|合法前缀轮先触沿|该组实际接近速度(m/s)|整段硬限位合法且轮先触沿|','|---|---|---|---|---|']
    for group in sorted(set((r['direction'],r['guide']) for r in dyn)):
        a=[r for r in dyn if (r['direction'],r['guide'])==group];sp=[r['actual_approach_speed_m_s'] for r in a if r['actual_approach_speed_m_s'] is not None]
        lines.append(f'|{group[0]}/{group[1]}|{sum(r["entered_s"] is not None for r in a)}/6|{sum(r["primary_wheel_first_legal_to_contact"] for r in a)}/6|{(f"{min(sp):.3f}～{max(sp):.3f}" if sp else "未接近台沿")}|{sum(r["whole_attempt_hard_limits_legal"] and r["primary_wheel_first_legal_to_contact"] for r in a)}/6|')
    lines+=['',
        '下表重点列正向高度匹配组。“合法前缀”仅指硬关节未越限，动态闭链误差另列；截止第一次硬限位超过1e−6rad之前；下一物理步检测到越限立即终止样本，保留原值，不夹紧关节掩盖。首次触沿都发生在越限之前，因此入口判据成立；整段尝试全部不计合法通过。其余24例详见[接触顺序与终止表JSON](dynamic-final/metrics.json)／[30例CSV](dynamic-final/contact-order.csv)。','',
        '|指令(m/s)/偏角(°)|进入时刻(s)/轮轴距沿(m)|首接触/时刻(s)|触沿前最小上棱距离(mm)|移动中负前伸最长(s)|实际接近速度(m/s)|压立面时轮轴最多升高左/右(mm)|首越限时刻(s)|','|---|---|---|---|---|---|---|---|']
    for r in hm:
        lines.append(f'|{r["speed"]:.1f}/{r["angle"]:+.0f}|{r["entered_s"]:.2f}/{r["entered_axle_distance_m"]:.3f}|{r["first_contact"]}/{r["first_contact_s"]:.3f}|{r["precontact_upper_edge_distance_min_m"]*1000:.2f}|{r["negative_overhang_moving_longest_s"]:.3f}|{r["actual_approach_speed_m_s"]:.3f}|{v([None if z is None else z*1000 for z in r["wall_loaded_rise_max_m"]],1)}|{r["first_limit_stop_s"]:.3f}|')
    lines+=['',
        '实际速度穿过.2～.5m/s区间时，负前伸可连续保持约.40～.57s，但属于加速过程，不是稳态低速跟踪。高度匹配组压立面时轮轴只升高约1.3～10.1mm，没有达到本轮定义的明显爬升（≥25mm）。没有完成登台；本轮也不以登台作为主判据。指令.2/.3/.5不等于实际速度，不能把.61～.76m/s的结果写成“实速.2～.5m/s通过”。','',
        '![正向高度匹配动态曲线](plots/forward-height_matched-dynamic.png)','',
        '![正向固定参考动态曲线](plots/forward-held-dynamic.png)','',
        '![倒车固定参考动态曲线](plots/reverse-held-dynamic.png)','',
        '## 数据、验证与复现','',
        '静态全程0积分。动态力/接触顺序/实际力矩/限位逐1ms记录，CPU运动学的前伸量/上棱距离曲线每2ms计算，并补齐首接触前后准确1ms点；对轮先触沿的7例，另逐1ms复查首接触前全部非轮部件到上棱的最小距离，见dynamic-final/contact-distance-1ms-audit.json。轮轴CPU/GPU位置逐点配对，最大误差<.02mm；静态缓存凸包与上轮原函数误差<1e−9m，重力虚功通过有限差分复算，6电机实际矩均在40/3.9Nm内。','',
        f'动态合法前缀中闭链残差最大{max(r["dynamic_closed_chain_residual_max_m"] for r in dyn)*1000:.4f}mm，反映原GPU约束的有限刚度，不能写成满足静态1e−7m残差。没有把触限后的样本、保护性复位、未达目标的倒车样本计入通过。','',
        '- `gpu/<方向-脚本>/traces/*.npz`：1ms求解前/积分后qpos、qvel、轮心、真实执行器力矩/ctrl、全部轮接触槽、机身/腿杆峰力位置、限位量、动作和有效标记。protocol.json记录模型哈希、延迟、入姿时刻和停机原因。',
        '- `dynamic-final/*-geometry.npz`：2ms实际前伸量、根高/pitch、各非轮部件到有限上棱线的距离、闭链残差和原始索引；每个NPZ与原始SHA对应。',
        '- 所有前缀/未入姿/触限失败仍留分母；全程0优化器更新，权重逐张量不变，容量无溢出。',
        '- [正常1倍速视频](video-final/approach-1x.mp4)仅展示合法前缀，在首越限前停帧，不展示越限后的运动冒充可行。原始记录仍保留首个超限值。','',
        '仓库根复现，OUT必须换为不存在的新目录；输出parent先建以便保存命令日志。以下合计30例，不额外启动训练或TensorBoard：','',
        '```bash','export PYTHONPATH=mjlab_training/src:.','PY=mjlab_training/.venv/bin/python','OUT=artifacts/fudan_ppo_step/新的轮先触沿实验',
        'mkdir "$OUT"',
        '$PY -m scripts.agent.fudan_ppo_step.local_overhang search --output "$OUT" --deadline "$(( $(date +%s) + 300 ))"',
        '$PY -m scripts.agent.fudan_ppo_step.local_overhang refine --output "$OUT" --deadline "$(( $(date +%s) + 180 ))"',
        'for B in 0 1 2 3 4; do',
        '  $PY -m scripts.agent.fudan_ppo_step.local_overhang run --output "$OUT" --batch "$B" --deadline "$(( $(date +%s) + 300 ))" || break','done',
        '$PY -m scripts.agent.fudan_ppo_step.local_overhang analyze --output "$OUT" --deadline "$(( $(date +%s) + 360 ))"',
        '$PY -m scripts.agent.fudan_ppo_step.local_overhang report --output "$OUT" --deadline "$(( $(date +%s) + 180 ))"',
        'MUJOCO_GL=egl $PY -m scripts.agent.fudan_ppo_step.local_overhang video --output "$OUT" --deadline "$(( $(date +%s) + 180 ))"','```','',
        '代码仅新增在scripts/agent/fudan_ppo_step/：local_overhang.py复用standing逆解并缓存mechanism凸包裁切；local_overhang_gpu.py复用StepEnv/原PD/ContactMonitor/冻结模型加载；local_overhang_report.py离线复算和出图。代码/依赖版本与哈希见code/manifest.json，不改原工具、任务、资产或Git。','',
        '下一步优先保留正向约.35～.36m姿态，单独解决高度/腿参考叠加后实际速度偏高的问题；先在平地验证实速能跟踪.2/.3/.5m/s且前伸量持续≤0，再接近台沿。若触沿后仍顶到膝上限，才设计独立收腿阶段；有效判据是原始1ms关节无越限、实际低速、轮先触沿，不能只看高度图。此项未继续执行，30例预算已用满。','']
    (out/'README.md').write_text('\n'.join(lines));print(json.dumps(summary,ensure_ascii=False),flush=True)
