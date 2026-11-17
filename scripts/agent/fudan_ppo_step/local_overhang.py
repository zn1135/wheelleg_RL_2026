"""Local hard-limit wheel-first search and bounded frozen3000 guidance only.

Reuses the standing closed-chain solver and reference mechanism hull clipping.
Search integrates no physics. Motor gravity loads include ground reactions and
closed-chain virtual work, with any required base holding moment stated apart.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from ct_mjlab.envs.chuanliantui.model import load_model,solve_standing_joint_positions,closed_chain_residual
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force
from .diagnose_reverse import write,sha
from scripts.agent.fudan_reference_step.mechanism import overhang,mesh_points

MOTORS=('lf0','lf00','lfwheel','rf0','rf00','rfwheel')
CAPS=np.array([40.,40.,3.9,40.,40.,3.9])
ROOT=Path(__file__).resolve().parents[3]


class Hulls:
    """Same convex-hull clipping as mechanism.overhang; cache invariant mesh edges."""
    def __init__(self,m):
        self.m=m;self.items=[];self.wids=[m.body(n).id for n in ('lfwheel','rfwheel')]
        for g in range(m.ngeom):
            if not m.geom_bodyid[g] or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
            assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
            mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy()
            h=ConvexHull(v);v=v[h.vertices];tri=ConvexHull(v).simplices
            edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[0,2]]]),axis=1),axis=0)
            self.items.append((g,int(m.geom_bodyid[g]),v,edges))

    def measure(self,d):
        wheel=[-np.inf,-np.inf];parts=[];lowest=np.inf
        for g,b,v,edges in self.items:
            p=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            if b in self.wids:
                wheel[0]=max(wheel[0],float(p[:,0].max()));wheel[1]=max(wheel[1],float(-p[:,0].min()));continue
            lowest=min(lowest,float(p[:,2].min()))
            below=p[p[:,2]<.2]
            if not len(below):continue
            a,bv=p[edges[:,0]],p[edges[:,1]];cross=(a[:,2]-.2)*(bv[:,2]-.2)<0;a,bv=a[cross],bv[cross]
            if len(a):below=np.r_[below,a+((.2-a[:,2])/(bv[:,2]-a[:,2]))[:,None]*(bv-a)]
            parts.append((self.m.body(b).name,float(below[:,0].max()),float(-below[:,0].min()),float(p[:,2].min())))
        if not parts:return [-1.,-1.],['none_below_200mm']*2,lowest,[]
        best=[max(parts,key=lambda a:a[1+i]) for i in (0,1)]
        return [best[i][1+i]-wheel[i] for i in (0,1)],[r[0] for r in best],lowest,parts


def jacobian(m,d):
    a=np.empty((3,m.nv));b=a.copy();r=[]
    pairs=[('left_rear_pin1','left_front_pin1'),('left_rear_pin2','left_front_pin2'),
           ('right_rear_pin1','right_front_pin1'),('right_rear_pin2','right_front_pin2'),
           ('left_spring_tip','left_gas_spring_lower'),('right_spring_tip','right_gas_spring_lower')]
    for x,y in pairs:
        mujoco.mj_jacSite(m,d,a,None,m.site(x).id);mujoco.mj_jacSite(m,d,b,None,m.site(y).id);r.append(a-b)
    return np.concatenate(r)


def reduction(m,d):
    motor_v=np.array([int(m.joint(n).dofadr[0]) for n in MOTORS])
    passive=np.array([v for v in range(6,m.nv) if v not in motor_v])
    J=jacobian(m,d);T=np.zeros((m.nv,6));T[motor_v]=np.eye(6)
    T[passive]=-np.linalg.lstsq(J[:,passive],J[:,motor_v],rcond=1e-10)[0]
    assert np.max(abs(J@T))<1e-7
    return T


def statics(m,d,T):
    mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d)
    gravity=np.empty(m.nv);mujoco.mj_rne(m,d,0,gravity)
    com=(d.xipos*m.body_mass[:,None]).sum(0)/m.body_mass.sum()
    wheels=d.xpos[[m.body(n).id for n in ('lfwheel','rfwheel')]]
    normals=m.body_mass.sum()*9.81*np.array([(com[1]-wheels[1,1])/(wheels[0,1]-wheels[1,1]),(wheels[0,1]-com[1])/(wheels[0,1]-wheels[1,1])])
    support=np.zeros(m.nv);jac=np.empty((3,m.nv))
    for n,F in zip(('lfwheel','rfwheel'),normals):
        bid=m.body(n).id;point=d.xpos[bid]-[0,0,.06]
        mujoco.mj_jac(m,d,jac,None,point,bid);support+=jac.T@np.array([0,0,F])
    spring=np.zeros(m.nv)
    for side in ('left','right'):
        j=m.joint(side+'_spring2_joint');v=int(j.dofadr[0]);q=int(j.qposadr[0])
        spring[v]=gas_spring_force(np.array(d.qpos[q]),np.array(0.)).item()
    tau_g=T.T@(gravity-support);tau_with_spring=T.T@(gravity-support-spring)
    return com,com-wheels.mean(0),gravity,tau_g,tau_with_spring,normals


def search(args):
    out=args.output/'static';out.mkdir(parents=True,exist_ok=False)
    m=load_model();d=mujoco.MjData(m);hulls=Hulls(m)
    baseline=solve_standing_joint_positions(m);klo=max(float(m.joint('lf1').range[0]),-float(m.joint('rf1').range[1]));khi=min(float(m.joint('lf1').range[1]),-float(m.joint('rf1').range[0]))
    kgrid=np.unique(np.r_[np.arange(klo,khi+.0001,args.knee_step),khi])
    hips=np.linspace(-math.pi,math.pi,round(360/args.hip_step)+1)[:-1]+baseline['lf0']
    pitches=np.arange(-25,25+.0001,args.pitch_step)
    rows=[];poses=[];gravity=[];maps=[];checks=[];tried=0;start=time.perf_counter()
    baseq=m.qpos0.copy()
    for k in kgrid:
        if time.time()>=args.deadline:raise TimeoutError('Static search wall deadline')
        js=solve_standing_joint_positions(m,front_joint_targets={'left':(.06,float(k)),'right':(-.06,float(-k))})
        for n,v in js.items():baseq[int(m.joint(n).qposadr[0])]=v
        d.qpos[:]=baseq;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);T=reduction(m,d)
        for hip in hips:
            q=baseq.copy();delta=hip-js['lf0']
            for n,sign in (('lf0',1),('lf00',1),('rf0',-1),('rf00',-1)):q[int(m.joint(n).qposadr[0])]+=delta*sign
            for pitch in pitches:
                tried+=1;theta=math.radians(pitch)/2;q[:7]=[0,0,0,math.cos(theta),0,math.sin(theta),0]
                d.qpos[:]=q;mujoco.mj_kinematics(m,d)
                h=.06-d.xpos[hulls.wids,2].mean()
                if h<.30-1e-9:continue
                d.qpos[2]=h;mujoco.mj_kinematics(m,d)
                limit=max([max(m.jnt_range[j,0]-d.qpos[m.jnt_qposadr[j]],d.qpos[m.jnt_qposadr[j]]-m.jnt_range[j,1],0.) for j in range(1,m.njnt) if m.jnt_limited[j]])
                closure=closed_chain_residual(m,d);wheelerr=abs(d.xpos[hulls.wids,2]-.06).max()
                if limit>1e-9 or closure>=1e-7 or wheelerr>=.0005:continue
                ov,part,low,components=hulls.measure(d)
                if low<0:continue
                com,offset,g,tg,tn,normal=statics(m,d,T)
                idx=len(rows)
                row=dict(index=idx,hip_rad=float(hip),knee_rad=float(k),pitch_deg=float(pitch),root_height_m=float(h),
                    overhang_m=ov,limiting_body=part,nonwheel_min_z_m=low,COM_world_m=com.tolist(),COM_minus_wheel_mid_m=offset.tolist(),
                    closed_residual_m=float(closure),wheel_target_error_m=float(wheelerr),hard_limit_excess_rad=float(limit),
                    loaded_gravity_motor_Nm=tg.tolist(),spring_compensated_motor_Nm=tn.tolist(),
                    motor_load_ratio=float((abs(tn)/CAPS).max()),normal_load_N=normal.tolist(),
                    base_holding_pitch_moment_Nm=float(m.body_mass.sum()*9.81*offset[0]),
                    motor_positions=[float(d.qpos[int(m.joint(n).qposadr[0])]) for n in MOTORS])
                rows.append(row);poses.append(d.qpos.copy());gravity.append(g);maps.append(T)
                if idx%2000==0:
                    exact=[overhang(m,d,hulls.wids,s)['limiting_component'] for s in (1,-1)]
                    err=max(abs(exact[i]['overhang_m']-ov[i]) for i in (0,1));assert err<1e-9;checks.append(dict(index=idx,cached_vs_original_error_m=err))
        print(json.dumps(dict(knee=float(k),valid=len(rows),wall_s=time.perf_counter()-start)),flush=True)
    assert rows
    np.savez_compressed(out/'search.npz',qpos=np.array(poses),tree_gravity_generalized=np.array(gravity),closed_tangent_map=np.array(maps))
    write(out/'search.json',dict(rows=rows,geometry_crosschecks=checks,grid=dict(knee_rad=kgrid.tolist(),hip_rad=hips.tolist(),pitch_deg=pitches.tolist()),
        tried=tried,physics_steps=0,optimizer_updates=0,wall_s=time.perf_counter()-start,
        joint_names=[m.joint(j).name for j in range(m.njnt)],motor_order=MOTORS,motor_caps_Nm=CAPS.tolist(),
        assets_sha256=sha(ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'),
        scope='Symmetric both wheels grounded, includes full2pi hip swing. Each knee closed once then both coaxial inputs rotated together; closure checked every accepted pose.'))
    mujoco.mj_saveModel(m,str(out/'model.mjb'))
    for i,n in enumerate(('forward','reverse')):
        best=min(rows,key=lambda r:r['overhang_m'][i]);print('BEST '+n+' '+json.dumps(best),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['search','refine','report','run','analyze','video'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--knee-step',type=float,default=.02);p.add_argument('--hip-step',type=float,default=3.);p.add_argument('--pitch-step',type=float,default=2.5)
    p.add_argument('--batch',type=int,default=0);a=p.parse_args()
    if a.mode=='search':search(a)
    elif a.mode=='refine':refine(a)
    elif a.mode=='report':report(a)
    elif a.mode=='run':run(a)
    elif a.mode=='analyze':analyze(a)
    elif a.mode=='video':video(a)



def refine(args):
    from scipy.optimize import least_squares,minimize
    out=args.output;m=mujoco.MjModel.from_binary_path(str(out/'static/model.mjb'));d=mujoco.MjData(m);hulls=Hulls(m)
    rows=json.loads((out/'static/search.json').read_text())['rows'];base=solve_standing_joint_positions(m);cache={}
    def evaluate(x):
        hip,k,pitch=map(float,x);key=round(k,12)
        if key not in cache:cache[key]=solve_standing_joint_positions(m,front_joint_targets={'left':(.06,k),'right':(-.06,-k)})
        js=cache[key];q=m.qpos0.copy()
        for n,v in js.items():q[int(m.joint(n).qposadr[0])]=v
        delta=hip-js['lf0']
        for n,sign in (('lf0',1),('lf00',1),('rf0',-1),('rf00',-1)):q[int(m.joint(n).qposadr[0])]+=sign*delta
        theta=math.radians(pitch)/2;q[:7]=[0,0,0,math.cos(theta),0,math.sin(theta),0]
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);d.qpos[2]=.06-d.xpos[hulls.wids,2].mean();mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        T=reduction(m,d);ov,parts,low,components=hulls.measure(d);com,offset,g,tg,tn,ns=statics(m,d,T)
        limits=[max(m.jnt_range[j,0]-d.qpos[m.jnt_qposadr[j]],d.qpos[m.jnt_qposadr[j]]-m.jnt_range[j,1],0.) for j in range(1,m.njnt) if m.jnt_limited[j]]
        return dict(root_height_m=float(d.qpos[2]),pitch_deg=pitch,hip_rad=hip,knee_rad=k,overhang_m=ov,limiting_body=parts,
            nonwheel_min_z_m=low,COM_minus_wheel_mid_m=offset.tolist(),loaded_gravity_motor_Nm=tg.tolist(),spring_compensated_motor_Nm=tn.tolist(),
            base_holding_pitch_moment_Nm=float(m.body_mass.sum()*9.81*offset[0]),motor_load_ratio=float((abs(tn)/CAPS).max()),
            closed_residual_m=float(closed_chain_residual(m,d)),hard_limit_excess_rad=float(max(limits)),
            wheel_target_error_m=float(abs(d.xpos[hulls.wids,2]-.06).max()),motor_positions=[float(d.qpos[int(m.joint(n).qposadr[0])]) for n in MOTORS],qpos=d.qpos.copy())
    fit=least_squares(lambda x:np.array([evaluate([x[0],x[1],0])['root_height_m']-.36,evaluate([x[0],x[1],0])['COM_minus_wheel_mid_m'][0]]),
        [-.65,.04],bounds=([-1.5,-.3],[.3,.13]),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100)
    f=evaluate([*fit.x,0]);f.update(direction='forward',selection='root .36m, pitch0, COM aligned with wheel midpoint')
    rev=[r for r in rows if r['overhang_m'][1]<-.005 and r['motor_load_ratio']<1]
    seed=min(rev,key=lambda r:abs(r['COM_minus_wheel_mid_m'][0]))
    constraints=[{'type':'ineq','fun':lambda x:evaluate(x)['root_height_m']-.30},
        {'type':'ineq','fun':lambda x:-.005-evaluate(x)['overhang_m'][1]}]
    opt=minimize(lambda x:abs(evaluate(x)['COM_minus_wheel_mid_m'][0]),[seed['hip_rad'],min(seed['knee_rad'],.12),seed['pitch_deg']],
        method='SLSQP',bounds=[(-1.5,.4),(-.34,.12),(-25,25)],constraints=constraints,options={'maxiter':100,'ftol':1e-10})
    r=evaluate(opt.x);r.update(direction='reverse',selection='Minimize COM offset with >=5mm wheel lead and knee margin>=20mrad',optimizer_success=bool(opt.success))
    selected=[f,r]
    for v in selected:
        i=0 if v['direction']=='forward' else 1
        assert v['overhang_m'][i]<=0 and v['closed_residual_m']<1e-7 and v['hard_limit_excess_rad']<=1e-9 and v['wheel_target_error_m']<.0005 and v['motor_load_ratio']<1
    np.savez_compressed(out/'selected-poses.npz',qpos=np.array([v.pop('qpos') for v in selected]))
    write(out/'selected.json',dict(rows=selected,physics_steps=0,optimizer_updates=0,
        direction_gate='Both directions have symmetric geometric solutions; no asymmetric search required',
        holding_torque_note='Closed-chain projected gravity minus vertical wheel load (mg split by COM-Y), with/without nominal springs; base held against nonzero COM-X moment. Not a free-standing equilibrium proof.'))
    print(json.dumps(selected,ensure_ascii=False),flush=True)


def run(args):
    from .local_overhang_gpu import run as action
    return action(args)


def analyze(args):
    from .local_overhang_gpu import analyze as action
    return action(args)


def video(args):
    from .local_overhang_gpu import video as action
    return action(args)


def report(args):
    from .local_overhang_report import report as action
    return action(args)


if __name__=='__main__':main()
