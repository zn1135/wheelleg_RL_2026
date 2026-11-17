"""Replay the actual Fudan parallel robot and ONNX on Warp GPU; never trains.

Upstream code is loaded without its desktop keyboard import. The controller,
observation/history and virtual-leg mapping methods remain the upstream methods.
All experimental scene edits occur in copies in a fresh artifacts directory.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import types
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REV = '8204e853dfd2ed06d85a322e1a998c3d20a3be2c'
XML_REL = 'mujoco/assert_now/infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml'
RT_REL = 'mujoco/python_tools/onnx_mj_binglian.py'
ONNX_REL = 'mujoco/actor/yuntai/上台阶3_angz+.onnx'
LOCAL = Path('/home/zn/文档/fudan_rl_wheel_leg-main')
FROZEN = ROOT/'artifacts/fudan_ppo_walk/fudan-ppo-walk-20261101/reference'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def fetch(rel):
    return urllib.request.urlopen('https://raw.githubusercontent.com/yly-true/fudan_rl_wheel_leg/'+REV+'/'+urllib.parse.quote(rel), timeout=45).read()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    tree = json.loads((FROZEN/'github-tree.json').read_text())['tree']
    index = {r['path']: r['sha'] for r in tree if r['type']=='blob'}
    wanted = [XML_REL, RT_REL, ONNX_REL]
    xmlbytes = fetch(XML_REL)
    root = ET.fromstring(xmlbytes)
    import posixpath
    for n in root.findall('./asset/*'):
        if 'file' in n.attrib:
            wanted.append(posixpath.normpath(posixpath.join(str(Path(XML_REL).parent), n.get('file'))))
    manifests = {}
    for rel in sorted(set(wanted)):
        data = None
        for base in (FROZEN, LOCAL):
            p=base/rel
            if p.is_file():
                b=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
                if blob==index.get(rel): data=b;break
        if data is None:data=xmlbytes if rel==XML_REL else fetch(rel)
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=index[rel]:raise ValueError('Git blob SHA mismatch: '+rel)
        p=out/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        manifests[rel]=dict(git_blob=blob,sha256=sha(p),bytes=len(data))
    write(out/'source-manifest.json',dict(repository='yly-true/fudan_rl_wheel_leg',commit=REV,files=manifests))
    print(json.dumps(dict(prepared=str(out),files=len(manifests))),flush=True)


def module(out):
    path=out/'source'/RT_REL
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='pynput')]
    mod=types.ModuleType('fudan_step_upstream');mod.__file__=str(path);sys.modules[mod.__name__]=mod
    exec(compile(tree,str(path),'exec'),mod.__dict__)
    # Limit ORT threads only; exact model graph and CPU execution retained.
    original=mod.load_onnx_session
    def load(path,device):
        opts=mod.ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        return mod.ort.InferenceSession(path,sess_options=opts,providers=['CPUExecutionProvider'])
    mod.load_onnx_session=load
    return mod


def scene(out, kind, height, width):
    src=out/'source'/XML_REL
    tree=ET.parse(src);root=tree.getroot()
    for n in root.findall('./asset/*'):
        if n.get('file'):n.set('file',str((src.parent/n.get('file')).resolve()))
    wb=root.find('worldbody')
    if kind=='single':
        for n in list(wb):
            if n.tag=='geom' and n.get('name')!='floor':wb.remove(n)
        if height>0:ET.SubElement(wb,'geom',name='research_platform',type='box',pos=f'{2.09/2} 0 {height/2}',
            size=f'{2.09/2} {width/2} {height/2}',rgba='.8 .5 .15 1',contype='2',conaffinity='1')
    target=out/'scenes'/f'{kind}_{height:.3f}_{width:.3f}.xml';target.parent.mkdir(exist_ok=True)
    if not target.exists():tree.write(target,encoding='unicode')
    return target


def clone_runtime(template, ref, args, state):
    rt=object.__new__(ref.BinglianRuntime)
    rt.__dict__.update(template.__dict__)
    rt.d=mujoco.MjData(rt.m)
    rt.command_state=state;rt.args=args
    rt.history=None;rt.last_actions=np.zeros(6,np.float32)
    rt.vel_last_pos=None;rt.vel_last_time=None
    rt.sensor_cache=dict(template.sensor_cache)
    return rt


def execute(out, label, specs, *, kind='single', height=.2, width=8., policy_dt=.004, seconds=9., deadline=None, friction=None,
            wheel_cap=5., delay_ms=0., probe=False):
    import torch
    import warp as wp
    import mujoco_warp as mw
    directory=out/'runs'/label;directory.mkdir(parents=True,exist_ok=False)
    ref=module(out);args=ref.build_arg_parser().parse_args([])
    args.xml=str(scene(out,kind,height,width));args.onnx=str(out/'source'/ONNX_REL);args.device='cpu'
    args.sim_dt=.002;args.policy_dt=policy_dt
    start=time.perf_counter();template=ref.BinglianRuntime(args,ref.CommandState())
    model=template.m;dt=float(model.opt.timestep);every=round(policy_dt/dt)
    if abs(every*dt-policy_dt)>1e-9:raise ValueError('Non-integral period')
    if friction is not None:model.geom_friction[:,0]=friction
    robot_bodies=np.array([i for i in range(1,model.nbody)])
    wbids=np.array([model.body(n).id for n in ('l_wheel_Link','r_wheel_Link')])
    base=model.body('base_Link_del').id
    robot_geoms=np.where(model.geom_bodyid>0)[0]
    leg_bodies=[i for i in robot_bodies if i not in [base,*wbids]]
    initial_q=model.qpos0.copy()
    runtimes=[];init=[]
    for spec in specs:
        st=ref.CommandState();st.cmd_height=spec.get('cmd_height',.27)
        rt=clone_runtime(template,ref,args,st)
        if kind=='single' or spec.get('relocate'):
            yaw=math.radians(spec.get('angle',0));xy=spec.get('spawn',[-.9,0.])
            rt.d.qpos[:]=initial_q;rt.d.qpos[:3]=[*xy,.21+spec.get('spawn_z',0.)]
            rt.d.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        mujoco.mj_forward(model,rt.d)
        init.append(rt.d.qpos.copy());runtimes.append(rt)
    N=len(specs);steps=round(seconds/dt)
    fields=('qpos','qvel','xpos','xquat','sensordata')
    acts=np.zeros((N,6),np.float32);qs=[None]*N;qds=[None]*N
    frozen_legs=[None]*N
    delay_steps=int(math.ceil(delay_ms/1000/dt-1e-9))
    sensor_queue=[];action_queue=[]
    recorder=None
    if probe:
        from .mechanism import ContactRecorder
        recorder=ContactRecorder(model,N,wbids)
    # Explicitly store every physics tick, not only policy frames.
    traces={k:[] for k in ('time','qpos','qvel','wheel_pos','wheel_force','wheel_support','base_force','leg_force','base_contact_pos','leg_contact_pos','wheel_side_force','motor_torque','virtual_q','actions','command','pitch_deg','roll_deg','virtual_torque')}
    errors=[None]*N;stopped=np.zeros(N,bool);complete=np.ones(N,bool)
    motor_ids=np.array([template.handles.lf0,template.handles.lw,template.handles.rf0,template.handles.rw,template.handles.lf20,template.handles.rf20])
    meta=dict(kind=kind,height=height,width=width,physics_dt=dt,pd_dt=dt,policy_dt=policy_dt,
        physics_backend='mujoco_warp_gpu',onnx_provider='CPUExecutionProvider',mujoco=mujoco.__version__,
        source_commit=REV,onnx_sha256=sha(out/'source'/ONNX_REL),original_policy_dt=.004,
        original_declared_policy_dt=.01,original_declared_sim_dt=.005,
        p_gains=template.active_p_gains.tolist(),d_gains=template.active_d_gains.tolist(),
        default_q=template.active_default_dof_pos.tolist(),command_scale=template.active_command_scale.tolist(),
        action_scale=[.5,10.],delay='No added delay; original state/finite-difference sampling retained',
        friction=model.geom_friction.tolist(),geom_names=[model.geom(i).name for i in range(model.ngeom)],
        body_names=[model.body(i).name for i in range(model.nbody)],mass=float(model.body_subtreemass[base]),
        original_robot_preserved=True,optimizer_updates=0,specs=specs,ngeom=model.ngeom,legs_motor_cap=30.,wheel_motor_cap=5.)
    meta.update(wheel_motor_cap=wheel_cap,requested_delay_ms=delay_ms,effective_delay_ms=delay_steps*dt*1000,
        delay_note='Both sensor packet and action ZOH delayed; ceil to physical ticks; commands/history last_actions undelayed',
        action_delay_initial_value='zero',sensor_delay_initial_value='reset state held',detailed_probe=probe)
    write(directory/'metadata.json',meta)
    mujoco.mj_saveModel(model,str(directory/'model.mjb'))
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        gm=mw.put_model(model);gd=mw.put_data(model,template.d,nworld=N,nconmax=512,njmax=2048)
        gd.qpos.assign(np.asarray(init,np.float32));gd.qvel.zero_();gd.time.zero_();gd.ctrl.zero_()
        mw.forward(gm,gd)
        capacity=512*N
        contact_ids=wp.array(np.arange(capacity,dtype=np.int32),dtype=wp.int32)
        cf=wp.zeros(capacity,dtype=wp.spatial_vector)
        with wp.ScopedCapture() as capture:
            mw.step(gm,gd);mw.contact_force(gm,gd,contact_ids,True,cf)
        graph=capture.graph
        allzero=np.zeros((N,model.nv),np.float32)
        status_age=np.zeros(N)
        times=0
        for tick in range(steps):
            if deadline and time.time()>=deadline:
                complete[:]=False;errors=[e or 'wall_deadline' for e in errors];break
            arrays={f:getattr(gd,f).numpy() for f in fields}
            sensor_queue.append(arrays)
            sensed=sensor_queue[max(0,len(sensor_queue)-1-delay_steps)]
            if len(sensor_queue)>delay_steps+1:sensor_queue.pop(0)
            for i,rt in enumerate(runtimes):
                for f in fields:getattr(rt.d,f)[:]=sensed[f][i]
                rt.d.time=tick*dt
            if tick%every==0:
                obs=[];hist=[]
                for i,(rt,spec) in enumerate(zip(runtimes,specs)):
                    target=0. if tick*dt<1. else spec['speed']
                    speed=np.clip((tick*dt-1.)/.5,0,1)*target
                    if kind=='single' and np.min(arrays['xpos'][i,wbids,0])>.45:speed=0.
                    if stopped[i]:speed=0.
                    rt.command_state.cmd_lin_vel=float(speed);rt.command_state.cmd_yaw_vel=0.
                    a,g,q,qd=rt.build_state();qs[i],qds[i]=q,qd
                    ob=rt.build_obs(a,g,q,qd);rt.update_history(ob);obs.append(ob);hist.append(rt.history.reshape(-1))
                output=template.sess.run(None,{template.obs_name:np.asarray(obs,np.float32),template.hist_name:np.asarray(hist,np.float32)})[0]
                acts=np.clip(output,-ref.CLIP_ACTIONS,ref.CLIP_ACTIONS).astype(np.float32)
                for i,rt in enumerate(runtimes):
                    freeze=specs[i].get('freeze_leg_after_s')
                    if freeze is not None and tick*dt>=freeze:
                        if frozen_legs[i] is None:frozen_legs[i]=acts[i,[0,1,3,4]].copy()
                        acts[i,[0,1,3,4]]=frozen_legs[i]
                    rt.last_actions[:]=acts[i]
            else:
                for i,rt in enumerate(runtimes):_,_,qs[i],qds[i]=rt.build_state()
            action_queue.append(acts.copy())
            applied=(np.zeros_like(acts) if delay_steps and len(action_queue)<=delay_steps
                     else action_queue[max(0,len(action_queue)-1-delay_steps)])
            if len(action_queue)>delay_steps+1:action_queue.pop(0)
            controls=[]
            for i,rt in enumerate(runtimes):
                ctrl=rt.compute_control(applied[i],qs[i],qds[i]);rt.apply_ctrl(ctrl)
                rt.d.ctrl[[rt.handles.lw,rt.handles.rw]]=np.clip(rt.d.ctrl[[rt.handles.lw,rt.handles.rw]],-wheel_cap,wheel_cap)
                controls.append(rt.d.ctrl.copy())
            controls=np.asarray(controls,np.float32)
            if not np.isfinite(controls).all():raise FloatingPointError('Non-finite control')
            gd.ctrl.assign(controls);wp.capture_launch(graph)
            pos=gd.xpos.numpy();q=gd.qpos.numpy();vel=gd.qvel.numpy()
            count=int(gd.nacon.numpy()[0]);nefc=gd.nefc.numpy()
            if count>capacity or nefc.max()>2048:raise RuntimeError('Contact/constraint capacity overflow')
            pair=gd.contact.geom.numpy()[:count];world=gd.contact.worldid.numpy()[:count]
            forces=cf.numpy()[:count,:3];cpos=gd.contact.pos.numpy()[:count];frame=gd.contact.frame.numpy()[:count]
            if recorder is not None:
                recorder.record(gd,arrays,tick*dt,pair,world,forces,cpos,frame,motor_ids,applied)
            wf=np.zeros((N,2,3));support=np.zeros((N,2));side=np.zeros((N,2));bf=np.zeros(N);lf=np.zeros(N);bp=np.zeros((N,3));lp=bp.copy()
            for j in range(count):
                a,b=pair[j];w=int(world[j]);f=forces[j];mag=np.linalg.norm(f)
                ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
                if ba and bb:continue
                for body,sign in ((ba,-1),(bb,1)):
                    if body in wbids:
                        k=int(np.where(wbids==body)[0][0]);wf[w,k]+=sign*f
                        normal=frame[j,0] if frame.ndim==3 else frame[j,:3]
                        if abs(normal[2])>.7:support[w,k]+=mag
                        else:side[w,k]=max(side[w,k],mag)
                    if body==base and mag>bf[w]:bf[w]=mag;bp[w]=cpos[j]
                    if body in leg_bodies and mag>lf[w]:lf[w]=mag;lp[w]=cpos[j]
            quat=q[:,3:7];R=np.array([mujoco_rotation(v) for v in quat])
            pitch=np.rad2deg(np.arcsin(np.clip(-R[:,2,0],-1,1)));roll=np.rad2deg(np.arctan2(R[:,2,1],R[:,2,2]))
            # Derived geometry has the normal MuJoCo one-physics-step lag.
            item=dict(time=(tick+1)*dt,qpos=q.copy(),qvel=vel.copy(),wheel_pos=pos[:,wbids].copy(),
                wheel_force=wf,wheel_support=support,base_force=bf,leg_force=lf,base_contact_pos=bp,
                leg_contact_pos=lp,wheel_side_force=side,motor_torque=controls[:,motor_ids],
                virtual_q=np.asarray(qs),actions=acts.copy(),command=np.array([[r.command_state.cmd_lin_vel,0,r.command_state.cmd_height] for r in runtimes]),
                pitch_deg=pitch,roll_deg=roll,virtual_torque=np.array([rt.active_p_gains*(acts[i]*np.array([.5,.5,0,.5,.5,0])+rt.active_default_dof_pos-qs[i])+rt.active_d_gains*(acts[i]*np.array([0,0,10,0,0,10])-qds[i]) for i,rt in enumerate(runtimes)]))
            for k,v in item.items():traces[k].append(v)
            invalid=~np.isfinite(q).all(-1)|(np.abs(vel)>1e4).any(-1)
            if invalid.any():
                complete[:]=False
                errors=['nonfinite_or_runaway_state' if bad else 'batch_stopped_after_invalid_peer' for bad in invalid]
                break
            # Fallen worlds remain in denominator; continue for visible failure evidence.
            if tick%500==0:print(json.dumps(dict(label=label,step=tick,total=steps,wall=time.perf_counter()-start,root_x=q[:,0].round(2).tolist())),flush=True)
        for i,spec in enumerate(specs):
            trace={k:np.asarray(v)[:,i] if k!='time' else np.asarray(v) for k,v in traces.items()}
            trace.update(complete=np.array(bool(complete[i])),error=np.array(errors[i] or ''))
            name=spec.get('id',f'case_{i:03d}')
            np.savez_compressed(directory/(name+'.npz'),**trace)
            if recorder is not None:recorder.save(directory/(name+'-probe.npz'),i)
    write(directory/'execution.json',dict(physics_steps=len(traces['time']),worlds=N,wall_s=time.perf_counter()-start,
        complete=complete.tolist(),optimizer_updates=0,policy_forwards_per_world=(len(traces['time'])+every-1)//every))
    print('FINISHED '+label,flush=True)
    return bool(complete.all())


def mujoco_rotation(q):
    R=np.empty(9);mujoco.mju_quat2Mat(R,q);return R.reshape(3,3)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label');p.add_argument('--cases',type=Path);p.add_argument('--height',type=float,default=.2)
    p.add_argument('--width',type=float,default=8.);p.add_argument('--kind',choices=['single','original'],default='single')
    p.add_argument('--policy-dt',type=float,default=.004);p.add_argument('--seconds',type=float,default=9.)
    p.add_argument('--deadline',type=float);p.add_argument('--friction',type=float)
    a=p.parse_args()
    if a.mode=='prepare':return prepare(a.output)
    execute(a.output,a.label,json.loads(a.cases.read_text()),kind=a.kind,height=a.height,width=a.width,
        policy_dt=a.policy_dt,seconds=a.seconds,deadline=a.deadline,friction=a.friction)


if __name__=='__main__':main()
