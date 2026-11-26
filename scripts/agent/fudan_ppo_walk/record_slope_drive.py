"""冻结5700：1ms实际驱动力/轮速/接触滑移记录，沿用允许斜坡触地规则。"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

from . import inspect_critic_scan_stage as inspect
from . import inspect_regions_stage as regional
from . import inspect_randomization_4000 as frozen
from .run_torque_ablation import run_command

ROOT=frozen.ROOT
PYTHON=ROOT/'mjlab_training/.venv/bin/python'
MODULE='scripts.agent.fudan_ppo_walk.record_slope_drive'


class Recorder:
    def __init__(self,env,out,latent):
        import torch
        self.env,self.out,self.latent=env,out,latent
        self.rows=[];self.pre=None;self.slices={};self.flushed=False
        self.tick=0;self.updated=False;self.target=torch.zeros(env.num_envs,2,device=env.device)
        self.unclipped=self.target.clone();self.pd_error=0.;self.ctrl_error=0.
        m=env.env.sim.mj_model;a=env.action
        names=('lf0','lf00','lfwheel','rf0','rf00','rfwheel')
        self.motors=[m.actuator('robot/'+n+'_motor').id for n in names]
        self.wheels=[m.body('robot/'+n).id for n in ('lfwheel','rfwheel')]
        self.roots=[int(m.body_rootid[i]) for i in self.wheels]
        probe=env.env.scene['wheel_drive_probe']
        self.probe_order=[next(i for i,n in enumerate(probe.primary_names) if n.split('/')[-1]==name)
                          for name in ('lfwheel','rfwheel')]
        assert np.all(m.actuator_gainprm[self.motors,0]==1.)
        assert np.all(m.actuator_gear[self.motors,0]==1.)
        assert not a.wheel_motor.enabled and not a._wheel_speed_shaping
        assert a.cfg.pd_decimation==5 and env.cfg.sim.dt==.001
        assert env.cfg.normalization.command[0]==2.
        self.metadata=dict(motor_order=names,wheel_order=['lfwheel','rfwheel'],
            actuator_ids=self.motors,wheel_body_ids=self.wheels,
            motor_dof_indices=a._motor_v_adr.cpu().tolist(),
            wheel_effort_limit_Nm=a._effort_limits[a._wheel_action_ids].cpu().tolist(),
            pd_dt_s=.005,physics_dt_s=.001,policy_dt_s=.01,
            encoder_velocity_scale=2.,
            pre_fields='Before integration, after actual controls have been written',
            post_fields='After integration / native contact sensors from this physical step',
            contact_slip_phase='Native contact point and cvel, normally solver-input/preintegration kinematics; CPU pairing audited',
            contact_reduction='Strongest contact on each wheel with terrain',
            saturation_definition='abs(actual wheel joint torque)>=0.99*3.9Nm',
            optimizer_updates=0)

    def pack(self,values):
        import torch
        chunks=[];offset=0
        for name,value in values:
            flat=value.reshape(self.env.num_envs,-1).float()
            self.slices[name]=[offset,offset+flat.shape[1]]
            offset+=flat.shape[1];chunks.append(flat)
        return torch.cat(chunks,-1)

    def after_apply(self,previous_tick):
        import torch
        if not self.env.action.record_substeps:return
        a=self.env.action;self.updated=previous_tick==0
        if self.updated:
            delayed=a._action_history[(a._history_head-a.wheel_action_delay_steps)%len(a._action_history),a._env_ids]
            w=a._wheel_action_ids
            self.target.copy_((a.cfg.wheel_velocity_scale*delayed[:,w]).clamp(-a.cfg.max_wheel_velocity,a.cfg.max_wheel_velocity))
            self.unclipped.copy_(a.kd_scale[:,w]*(a.cfg.wheel_stiffness*(self.target-a._env.sim.data.qvel[:,a._motor_v_adr[w]])
                -a.cfg.wheel_damping*a._env.sim.data.qvel[:,a._motor_v_adr[w]])*a.motor_strength[:,w])
            expected=self.unclipped.clamp(-a._effort_limits[w],a._effort_limits[w])
            self.pd_error=max(self.pd_error,float((expected-a.torques[:,w]).abs().max()))

    def before_step(self):
        import torch
        e=self.env;a=e.action;s=e.env.sim
        if not a.record_substeps:return
        ctrl=s.data.ctrl[:,self.motors]
        self.ctrl_error=max(self.ctrl_error,float((ctrl-a.torques).abs().max()))
        # 这些都是仿真实际当拍数据，不从网络输出反推施加力矩。
        self.pre_values=[('qpos_pre',s.data.qpos.clone()),('qvel_pre',s.data.qvel.clone()),
            ('motor_qvel_pre',s.data.qvel[:,a._motor_v_adr].clone()),
            ('raw_policy_action',a.raw_action.clone()),('delayed_action',a._control_actions.clone()),
            ('wheel_target_rad_s',self.target.clone()),('wheel_unclipped_torque_Nm',self.unclipped.clone()),
            ('pd_torque_Nm',a.torques.clone()),('actual_ctrl_Nm',ctrl.clone()),
            ('command',e.commands.clone()),('encoder_raw',self.latent[0].clone()),
            ('encoder_reference_velocity',e.linear.clone()),
            ('pd_updated',torch.full((e.num_envs,1),float(self.updated),device=e.device))]

    def after_step(self):
        import torch
        e=self.env;a=e.action;s=e.env.sim
        if not a.record_substeps:return
        probe=e.env.scene['wheel_drive_probe'].data;o=self.probe_order
        point=probe.pos[:,o];normal=probe.normal[:,o]
        cvel=s.data.cvel[:,self.wheels];center=s.data.subtree_com[:,self.roots]
        contact_velocity=cvel[...,3:]+torch.cross(cvel[...,:3],point-center,dim=-1)
        tangential=contact_velocity-(contact_velocity*normal).sum(-1,keepdim=True)*normal
        found=probe.found[:,o].reshape(e.num_envs,2)>0
        tangential=torch.where(found[...,None],tangential,torch.zeros_like(tangential))
        post=[('qpos_post',s.data.qpos.clone()),('qvel_post',s.data.qvel.clone()),
            ('actual_actuator_force_Nm',s.data.actuator_force[:,self.motors].clone()),
            ('actual_joint_torque_Nm',s.data.qfrc_actuator[:,a._motor_v_adr].clone()),
            ('motor_qvel_post',s.data.qvel[:,a._motor_v_adr].clone()),
            ('gpu_row_post',a.substep_rows[-1].clone()),
            ('wheel_net_force_world_N',e.env.scene['wheel_contact'].data.force.clone()),
            ('contact_found',found.float()),('contact_point_world_m',point.clone()),
            ('contact_normal_world',normal.clone()),('contact_force_world_N',probe.force[:,o].clone()),
            ('contact_tangent_velocity_world_m_s',tangential),
            ('wheel_cvel',cvel.clone()),('wheel_subtree_com_world_m',center.clone())]
        self.rows.append(self.pack(self.pre_values+post));self.tick+=1

    def flush(self):
        import torch
        if self.flushed:return
        self.flushed=True
        self.out.mkdir(parents=True,exist_ok=False)
        if not self.rows:
            frozen.write(self.out/'recording-error.json',dict(error='No recorded physics steps'));return
        all_data=torch.stack(self.rows).cpu().numpy()
        samples=regional.routes()
        assert all_data.shape[1]==len(samples)==32
        for i,sample in enumerate(samples):
            count=min(len(all_data),round(sample['duration']/.001))
            data={name:all_data[:count,i,lo:hi].copy() for name,(lo,hi) in self.slices.items()}
            data['time_start_s']=np.arange(count)*.001
            data['time_end_s']=(np.arange(count)+1)*.001
            np.savez_compressed(self.out/(sample['id']+'.npz'),**data)
        self.metadata.update(physics_steps=self.tick,pd_formula_max_error_Nm=self.pd_error,
            ctrl_equals_limited_PD_max_error_Nm=self.ctrl_error,fields=self.slices,
            complete_expected_batch=self.tick==11000,samples=samples)
        frozen.write(self.out/'metadata.json',self.metadata)


def worker(out):
    import torch
    inspect.configure('scan')
    proof=json.loads((out/'scan/checkpoint-audit.json').read_text())
    checkpoint=out/'scan/checkpoint/model_5700.pt'
    assert frozen.sha(checkpoint)==proof['checkpoint_sha256']
    assert not any(n=='ct_mjlab' or n.startswith('ct_mjlab.') for n in sys.modules)
    sys.meta_path.insert(0,frozen.FrozenTrainingSource())
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.fudan_ppo_walk import policy as loading
    from mjlab.sensor import ContactSensorCfg,ContactMatch
    old_cfg,old_init,old_sample,old_load=physical.make_env_cfg,physical.FudanEnv.__init__,physical.sample_substep,loading.load_policy
    recorders=[];latent=[None];networks=[]
    def configure(*args,**kwargs):
        cfg=old_cfg(*args,**kwargs);cfg.sim.nconmax,cfg.sim.njmax=64,256
        cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name='wheel_drive_probe',
            primary=ContactMatch(mode='body',pattern='^(lfwheel|rfwheel)$',entity='robot'),
            secondary=ContactMatch(mode='body',pattern='terrain'),
            fields=('found','force','pos','normal','tangent'),reduce='maxforce',global_frame=True))
        return cfg
    def initialized(env,*args,**kwargs):
        old_init(env,*args,**kwargs)
        rec=Recorder(env,out/'drive',latent);recorders.append(rec);env._drive_recorder=rec
        old_apply,old_step,old_close=env.action.apply_actions,env.env.sim.step,env.close
        def apply():
            tick=env.action._pd_tick;old_apply();rec.after_apply(tick)
        def step():
            rec.before_step();return old_step()
        def close():
            try:rec.flush()
            finally:old_close()
        env.action.apply_actions,env.env.sim.step,env.close=apply,step,close
    def sampled(env):
        value=old_sample(env)
        # 原采样/延迟队列先按原顺序更新，诊断只读本次实际状态。
        wrapper=getattr(env,'_drive_wrapper',None)
        if wrapper is not None:wrapper._drive_recorder.after_step()
        return value
    def loaded(*args,**kwargs):
        policy,meta=old_load(*args,**kwargs)
        before={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        networks.append((policy,before));act=policy.act_inference
        def inference(*args,**kwargs):
            result=act(*args,**kwargs);latent[0]=result[1].detach();return result
        policy.act_inference=inference
        return policy,meta
    # 安装包装器引用供每个1ms指标回调读取；不改环境或网络输入。
    base_initialized=initialized
    def initialized_with_ref(env,*args,**kwargs):
        base_initialized(env,*args,**kwargs);env.env._drive_wrapper=env
    destination=out/'probe';destination.mkdir()
    with patch.object(physical,'make_env_cfg',configure),patch.object(physical.FudanEnv,'__init__',initialized_with_ref), \
         patch.object(physical,'sample_substep',sampled),patch.object(loading,'load_policy',loaded), \
         patch('torch.optim.Adam.step',side_effect=AssertionError('Recording cannot learn')):
        regional.terrain(destination,5700,models=[('5700',checkpoint)])
    assert len(recorders)==1 and recorders[0].tick==11000
    assert recorders[0].pd_error<2e-5 and recorders[0].ctrl_error<2e-5
    assert all(torch.equal(v.cpu(),before[k]) for policy,before in networks for k,v in policy.state_dict().items())
    frozen.write(out/'runtime-audit.json',dict(passed=True,physics_steps=11000,num_envs=32,
        policy_unchanged=True,checkpoint_unchanged=frozen.sha(checkpoint)==proof['checkpoint_sha256'],
        pd_formula_max_error_Nm=recorders[0].pd_error,actual_ctrl_matches_PD=True,
        frozen_modules=frozen.LOADED,optimizer_updates=0))


def report(out,source):
    import mujoco
    meta=json.loads((out/'drive/metadata.json').read_text())
    assert meta['complete_expected_batch']
    model=mujoco.MjModel.from_binary_path(str(out/'probe/terrain/5700/fixtures/model.mjb'))
    data=mujoco.MjData(model);paired=0;diffs=[]
    old=source/'scan/terrain/5700';new=out/'probe/terrain/5700'
    for path in sorted((new/'fixtures').glob('*_initial.npz'))+[new/'initial.npz']:
        with np.load(path) as a,np.load(old/path.relative_to(new)) as b:
            for k in a.files:
                paired+=1
                if not np.array_equal(a[k],b[k]):diffs.append(dict(path=str(path.relative_to(new)),field=k))
    summaries={};kinematic_error=0.;force_error=0.
    wheels=[model.body('robot/'+n).id for n in ('lfwheel','rfwheel')]
    for sid in ('pyramid_3_up_v1','pyramid_3_up_v0.5','pyramid_2_up_v1','pyramid_0_up_v1'):
        with np.load(out/'drive'/(sid+'.npz')) as saved:d={k:saved[k].copy() for k in saved.files}
        assert all(np.isfinite(v).all() for v in d.values())
        force_error=max(force_error,float(np.abs(d['actual_joint_torque_Nm']-d['actual_ctrl_Nm']).max()))
        for k in (100,440,500,1000,2000):
            data.qpos[:]=d['qpos_pre'][k];data.qvel[:]=d['qvel_pre'][k]
            mujoco.mj_forward(model,data)
            for w,bid in enumerate(wheels):
                if not d['contact_found'][k,w]:continue
                velocity=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,velocity,0)
                point=d['contact_point_world_m'][k].reshape(2,3)[w]
                normal=d['contact_normal_world'][k].reshape(2,3)[w]
                point_velocity=velocity[3:]+np.cross(velocity[:3],point-data.xipos[bid])
                tangent=point_velocity-np.dot(point_velocity,normal)*normal
                actual=d['contact_tangent_velocity_world_m_s'][k].reshape(2,3)[w]
                kinematic_error=max(kinematic_error,float(np.abs(tangent-actual).max()))
        stats={}
        for lo,hi in ((0.,.4),(.45,1.),(1.,2.),(2.,4.)):
            mask=(d['time_start_s']>=lo)&(d['time_start_s']<hi)
            force=d['actual_joint_torque_Nm'][mask][:,[2,5]]
            slip=np.linalg.norm(d['contact_tangent_velocity_world_m_s'][mask].reshape(-1,2,3),axis=-1)
            found=d['contact_found'][mask]>0
            contact_force=d['contact_force_world_N'][mask].reshape(-1,2,3)
            normal=d['contact_normal_world'][mask].reshape(-1,2,3)
            normal_component=(contact_force*normal).sum(-1)
            normal_force=np.abs(normal_component)
            tangent_force=np.linalg.norm(contact_force-normal_component[...,None]*normal,axis=-1)
            loaded=found & (normal_force>5.)
            stats[f'{lo:g}-{hi:g}s']=dict(body_vx_mean_m_s=float(d['gpu_row_post'][mask,3].mean()),
                wheel_actual_torque_mean_Nm=force.mean(0).tolist(),wheel_actual_torque_abs_peak_Nm=np.abs(force).max(0).tolist(),
                wheel_saturation_fraction=(np.abs(force)>=.99*3.9).mean(0).tolist(),
                wheel_target_mean_rad_s=d['wheel_target_rad_s'][mask].mean(0).tolist(),
                wheel_actual_mean_rad_s=d['motor_qvel_post'][mask][:,[2,5]].mean(0).tolist(),
                contact_slip_mean_when_contact_m_s=[float(slip[found[:,w],w].mean()) if found[:,w].any() else None for w in (0,1)],
                loaded_contact_fraction=loaded.mean(0).tolist(),
                loaded_tangent_to_normal_force_ratio_median=[float(np.median(tangent_force[loaded[:,w],w]/normal_force[loaded[:,w],w]))
                    if loaded[:,w].any() else None for w in (0,1)],
                wheel_support_mean_N=(-d['wheel_net_force_world_N'][mask].reshape(-1,2,3)[:,:,2]).mean(0).tolist(),
                body_contact_mean_N=float(d['gpu_row_post'][mask,16].mean()),
                encoder_estimated_vx_mean_m_s=float((d['encoder_raw'][mask,0]/2).mean()),
                encoder_reference_vx_mean_m_s=float(d['encoder_reference_velocity'][mask,0].mean()))
        summaries[sid]=stats
        if sid=='pyramid_3_up_v1':
            names=['time_s','vx_command','vx_actual','body_contact_N','left_target_rad_s','right_target_rad_s',
                'left_actual_rad_s','right_actual_rad_s','left_actual_torque_Nm','right_actual_torque_Nm','left_slip_m_s','right_slip_m_s','estimated_vx_m_s']
            slip=np.linalg.norm(d['contact_tangent_velocity_world_m_s'].reshape(-1,2,3),axis=-1)
            values=np.column_stack((d['time_end_s'],d['command'][:,0],d['gpu_row_post'][:,3],d['gpu_row_post'][:,16],
                d['wheel_target_rad_s'],d['motor_qvel_post'][:,[2,5]],d['actual_joint_torque_Nm'][:,[2,5]],slip,d['encoder_raw'][:,0]/2))
            np.savetxt(out/'20deg-v1-height026.csv',values,delimiter=',',header=','.join(names),comments='')
    assert force_error<2e-5,force_error
    assert kinematic_error<2e-4,kinematic_error
    result=dict(passed=True,summaries=summaries,initial_arrays_compared=paired,initial_arrays_bitwise_equal=not diffs,
        initial_differences=diffs,actual_joint_force_matches_written_control_max_error_Nm=force_error,
        contact_velocity_CPU_geometry_validation_max_error_m_s=kinematic_error,
        nominal_wheel_friction=[float(model.geom_friction[next(i for i in range(model.ngeom)
            if model.geom_bodyid[i]==bid and model.geom_group[i]==3),0]) for bid in wheels],
        nominal_wheel_geom_priority=[int(model.geom_priority[next(i for i in range(model.ngeom)
            if model.geom_bodyid[i]==bid and model.geom_group[i]==3)]) for bid in wheels],
        physical_evaluation_backend='mujoco_warp_gpu',cpu_geometry_only=True,optimizer_updates=0)
    frozen.write(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','worker','report'))
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='worker':return worker(out)
    if a.mode=='report':return report(out,a.source.resolve())
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'recording-source.py')
    inspect.audit(out,5700,'scan')
    try:
        run_command(out,'record',[str(PYTHON),'-u','-m',MODULE,'worker','--output',str(out)])
        report(out,a.source.resolve())
        frozen.write(out/'status.json',dict(status='completed',optimizer_updates=0,training_started=False,finished_unix=time.time()))
    except BaseException as exc:
        frozen.write(out/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
