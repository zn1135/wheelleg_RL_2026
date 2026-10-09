"""Independent native MuJoCo playback using the same V14 policy interface."""
import json
import time
import numpy as np
import mujoco
import torch
from tensordict import TensorDict
from .asset import cpu_model, POSES
from .config import JOINTS, DT, DECIMATION, STEP_DT
from .recipe import Recipe

def rotation(quat):
    m=np.zeros(9);mujoco.mju_quat2Mat(m,quat);return m.reshape(3,3)

class CpuPlayback:
    def __init__(self):
        self.model=cpu_model();self.data=mujoco.MjData(self.model)
        m=self.model
        self.q=[m.jnt_qposadr[m.joint(n).id] for n in JOINTS]
        self.v=[m.jnt_dofadr[m.joint(n).id] for n in JOINTS]
        self.ctrl=[m.actuator(n+'_motor').id for n in JOINTS]
        self.gas=[m.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
        self.base=m.body('base_link').id;self.wheel=[m.body(n).id for n in JOINTS[4:]]
        nominal=torch.tensor([POSES['nominal'][n] for n in JOINTS],dtype=torch.float32)
        self.recipe=Recipe(1,'cpu',nominal,randomize=False)
        self.reset()
    def reset(self):
        m,d=self.model,self.data;mujoco.mj_resetData(m,d)
        d.qpos[:7]=(0,0,.15,1,0,0,0)
        for name,value in POSES['reset'].items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        d.ctrl[self.gas]=150;mujoco.mj_forward(m,d)
        self.steps=0;self.counter=0;self.contact_history=np.zeros((3,m.nbody))
        self.acc=np.zeros(6)
        self.recipe.reset(torch.tensor([0]));self.obs=self.observations()
    def contacts(self):
        forces=np.zeros((self.model.nbody,3));f=np.zeros(6)
        for i in range(self.data.ncon):
            c=self.data.contact[i];mujoco.mj_contactForce(self.model,self.data,i,f)
            world=c.frame.reshape(3,3).T@f[:3]
            a,b=self.model.geom_bodyid[[c.geom1,c.geom2]]
            forces[a]+=world;forces[b]-=world
        force=np.linalg.norm(forces,axis=-1)
        self.contact_history[1:]=self.contact_history[:-1].copy();self.contact_history[0]=force
        return force
    def state(self):
        m,d=self.model,self.data;rot=rotation(d.qpos[3:7]);vel=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.base,vel,0)
        lin=rot.T@vel[3:];ang=rot.T@vel[:3]
        wheel_vel=[]
        for bid in self.wheel:
            w=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,w,0)
            local=rot.T@(w[3:]-vel[3:]);local[1]=0;wheel_vel.append(local)
        wpos=(d.xpos[self.wheel]-d.xpos[self.base])@rot
        pitch=np.arcsin(np.clip(-rot[2,0],-1,1));roll=np.arctan2(rot[2,1],rot[2,2])
        bad=[i for i in range(1,m.nbody) if i not in self.wheel]
        peaks=self.contact_history.max(0)
        values=dict(pos=d.qpos[self.q],vel=d.qvel[self.v],acc=self.acc,torque=d.actuator_force[self.ctrl],
            lin=lin,ang=ang,gravity=rot.T@np.array([0,0,-1.]),height=d.qpos[2],pitch=pitch,roll=roll,
            wheel_pos=wpos,wheel_vel=np.array(wheel_vel),wheel_contact=peaks[self.wheel],
            bad_contact=peaks[bad].max(),base_contact=self.contact_history[0,self.base])
        return {k:torch.as_tensor(np.asarray(v).copy(),dtype=torch.float32).unsqueeze(0) for k,v in values.items()}
    def observations(self,advance=True):return TensorDict(self.recipe.observations(self.state(),advance),batch_size=[1])
    def step(self,actions):
        self.recipe.actions.copy_(actions.cpu());d=self.data
        for _ in range(DECIMATION):
            previous_vel=d.qvel[self.v].copy()
            tau=self.recipe.torque(torch.tensor(d.qpos[self.q],dtype=torch.float32)[None],
                torch.tensor(d.qvel[self.v],dtype=torch.float32)[None],torch.tensor([self.steps]))
            d.ctrl[self.ctrl]=tau.numpy()[0];d.ctrl[self.gas]=150
            mujoco.mj_step(self.model,d);self.contacts()
            self.acc=(d.qvel[self.v]-previous_vel)/DT
            mujoco.mj_forward(self.model,d)
            self.recipe.observe_substep(self.state())
        mujoco.mj_forward(self.model,d);self.steps+=1;s=self.state()
        immediate=not torch.isfinite(torch.cat((s['vel'],s['lin'],s['ang']),-1)).all()
        immediate|=bool((s['vel'].abs()>500).any()|(s['lin'].abs()>100).any()|(s['ang'].abs()>200).any())
        if self.steps>10:immediate|=bool(self.recipe.bad_obs[0]|(self.recipe.raw_obs_max[0]>120))
        bad=bool(s['base_contact'][0]>1 or abs(s['roll'][0])>np.deg2rad(40) or abs(s['pitch'][0])>np.deg2rad(40))
        self.counter=self.counter+1 if bad else 0
        done=self.counter>=20 or immediate
        terms=self.recipe.reward_terms(s,torch.tensor([done]));self.obs=self.observations(advance=False)
        return s,done,terms

def evaluate(actor,seconds=30,render=False,report=None):
    env=CpuPlayback();viewer=None
    if render:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(env.model,env.data)
    rows=[];first_failure=None
    try:
        for i in range(round(seconds/STEP_DT)):
            start=time.monotonic()
            with torch.inference_mode():action=actor(env.obs) if actor else torch.zeros(1,6)
            s,done,_=env.step(action)
            rows.append(dict(time=(i+1)*STEP_DT,height=float(s['height'][0]),pitch=float(s['pitch'][0]),
                speed=float(torch.linalg.vector_norm(s['lin'][0,:2])),x=float(env.data.qpos[0]),y=float(env.data.qpos[1])))
            if done and first_failure is None:first_failure=(i+1)*STEP_DT
            if not np.isfinite(env.data.qpos).all():break
            if viewer:
                if not viewer.is_running():break
                viewer.sync();time.sleep(max(0,STEP_DT-(time.monotonic()-start)))
    finally:
        if viewer:viewer.close()
    tail=rows[len(rows)//2:]
    metrics=dict(seconds=len(rows)*STEP_DT,first_failure_s=first_failure,
        max_height_error_last_half=max(abs(r['height']-.22) for r in tail),
        max_pitch_deg_last_half=max(abs(r['pitch'])*180/np.pi for r in tail),
        max_speed_last_half=max(r['speed'] for r in tail),
        max_drift=max(np.hypot(r['x'],r['y']) for r in rows))
    metrics['passed']=bool(first_failure is None and len(rows)*STEP_DT>=30 and
        metrics['max_height_error_last_half']<=.02 and metrics['max_pitch_deg_last_half']<=5 and
        metrics['max_speed_last_half']<=.05 and metrics['max_drift']<=.2)
    if report:
        path=__import__('pathlib').Path(report)
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(dict(metrics=metrics,trajectory=rows),indent=2)+'\n')
    return metrics
