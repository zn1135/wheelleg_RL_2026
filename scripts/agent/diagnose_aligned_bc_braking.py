#!/usr/bin/env python3
"""Train-only live student/teacher braking probes and actual takeover; no optimizer."""
import argparse,copy,json,math,shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from ct_mjlab.bc import load_actor,ALIGNED_LABEL_REFERENCE
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT
from ct_mjlab.envs.chuanliantui import multimode_observations as observations
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction
from ct_mjlab.teacher.standup import DONE
from check_lqr_student_closed_loop import digest,ROOT
from check_bc_local_feedback import frozen_state,pd_request,read_inputs,teacher_target
import collect_reference_bc as collection

ANCHORS=(8.,9.99,15.,20.,25.)
PERTURBATIONS=(('pitch_rad',math.radians(.5)),('wheel_forward_rad_s',.25),
               ('wheel_encoder_shift_m',.01),('lqr_position_integral_shift_m',.01))


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    actor,saved=load_actor(args.checkpoint,'cuda:0')
    assert saved['dataset']['observation_contract']==HANDOVER_OBSERVATION_CONTRACT
    assert saved['dataset']['label_reference']==ALIGNED_LABEL_REFERENCE
    immutable={str(p):digest(p) for p in (args.checkpoint,args.dataset,args.pose_dataset,
        ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')}
    before={k:v.clone() for k,v in actor.state_dict().items()}
    probes={};extras=[];branch_errors=[]
    class ProbedAction(collection.AlignedAction):
        def process_actions(self,actions):
            tick=getattr(self,'diagnostic_tick',0);self.diagnostic_tick=tick+1
            sample=tick in {round(t/.01) for t in ANCHORS}
            baseline=None
            if sample:
                history=copy.deepcopy(self.teacher.controller)
                obs=observations.get_state(self._env).actor().clone()
                q=self._env.sim.data.qpos[:,self._motor_q_adr].clone()
                dq=self._env.sim.data.qvel[:,self._motor_v_adr].clone()
                inputs={k:v.clone() for k,v in read_inputs(self).items()}
                inputs['position_reference']=observations.get_state(self._env).wheel_angle_handover_latched.clone()
                valid=(self._env._source_probe['live']&(self._env._source_probe['hold']>=500)&
                       (history.standup.phase==DONE)&~history.air.flight&self.balance_synchronized)
                assert valid.any()
                predicted=actor.mlp(obs)
                torch.testing.assert_close(predicted,actions,atol=2e-6,rtol=2e-6)
                baseline,desired=teacher_target(self,copy.deepcopy(history),inputs,q,dq)
                entry=dict(valid=valid.cpu().numpy(),observation=obs.cpu().numpy(),
                    physical_q=q.cpu().numpy(),physical_dq=dq.cpu().numpy(),
                    **{'base.teacher.target':baseline.cpu().numpy(),
                       'base.teacher.torque':pd_request(self,baseline,q,dq).cpu().numpy(),
                       'base.student.target':actions.cpu().numpy(),
                       'base.student.torque':pd_request(self,actions,q,dq).cpu().numpy()})
                for variable,amplitude in PERTURBATIONS:
                    for sign in (-1,1):
                        delta=sign*amplitude;altered=obs.clone();physical_dq=dq.clone();physical_q=q.clone()
                        measurement={k:v.clone() for k,v in inputs.items()};controller=copy.deepcopy(history)
                        if variable=='pitch_rad':
                            measurement['pitch']+=delta
                            measurement['upright']=measurement['roll'].cos()*measurement['pitch'].cos()
                            angle=torch.asin(obs[:,7].clamp(-1,1))+delta;roll=torch.atan2(-obs[:,8],-obs[:,9])
                            altered[:,7]=angle.sin();altered[:,8]=-roll.sin()*angle.cos();altered[:,9]=-roll.cos()*angle.cos()
                        elif variable=='wheel_forward_rad_s':
                            measurement['wheel_speed']+=delta
                            physical_dq[:,2]+=delta;physical_dq[:,5]-=delta
                            altered[:,20]+=delta*.1;altered[:,21]-=delta*.1
                        elif variable=='wheel_encoder_shift_m':
                            # 只改编码器位置；LQR速度融合与位置积分历史保持原样。
                            physical_q[:,2]+=delta/.06;physical_q[:,5]-=delta/.06
                            altered[:,14]+=delta/.06;altered[:,15]-=delta/.06
                        else:
                            # 这是内部参考反事实，不伪装成同一个物理传感器扰动。
                            controller.pos+=delta
                        target,_=teacher_target(self,controller,measurement,physical_q,physical_dq)
                        for name,value in (('teacher',target),('student',actor.mlp(altered))):
                            prefix=f'{variable}.{sign}.{name}'
                            entry[prefix+'.target']=value.cpu().numpy()
                            entry[prefix+'.torque']=pd_request(self,value,physical_q,physical_dq).cpu().numpy()
                np.savez_compressed(args.output/f'controller-{tick}.npz',**frozen_state(history),
                    **{'input.'+k:v.cpu().numpy() for k,v in inputs.items()})
            super().process_actions(actions)
            if baseline is not None:
                torch.testing.assert_close(baseline[valid],self.label[valid],atol=2e-6,rtol=2e-6)
                torch.testing.assert_close(desired[valid],self.teacher.torques[valid],atol=2e-6,rtol=2e-6)
                branch_errors.append(float(abs(baseline[valid]-self.label[valid]).max()))
                probes[tick]=entry
            if args.branch=='student':
                # 教师只旁路观察；所有真实动作仍来自冻结学生，禁止导出查询标签训练。
                self.selected.zero_();self.label_eligible.zero_()
                HuananhuPdAction.process_actions(self,actions)
            c=self.teacher.controller
            extras.append(dict(teacher_x=c.x.cpu().numpy().copy(),teacher_air=c.air.flight.cpu().numpy().copy(),
                teacher_yaw_target=c.yaw_target.cpu().numpy().copy(),
                teacher_wheel_contributions=c.contributions[:,:2].cpu().numpy().copy(),
                teacher_requested=self.teacher.torques.cpu().numpy().copy(),
                student_requested=pd_request(self,actions,self._source_q,self._source_dq).cpu().numpy().copy()))
    call=SimpleNamespace(base=args.dataset,pose_dataset=args.pose_dataset,mode='takeover',checkpoint=args.checkpoint,
        wheel_angle_slots=True,wheel_angle_handover_reset=True,legacy_zero_wheel=False,
        takeover_s=10.,duration_s=30.,output=args.output,device='cuda:0',fill_missing_groups=None,all_delay_pairs=False)
    with patch.object(collection,'AlignedAction',ProbedAction):collection.collect(call)
    path=args.output/'trace.npz'
    with np.load(path) as d:arrays={k:d[k] for k in d.files}
    arrays.update({k:np.asarray([x[k] for x in extras]) for k in extras[0]})
    np.savez_compressed(path,**arrays)
    archive={f'{tick}.{key}':value for tick,entry in probes.items() for key,value in entry.items()}
    np.savez_compressed(args.output/'probes.npz',**archive)
    result_path=args.output/'result.json';r=json.loads(result_path.read_text())
    assert r['base_motion_fingerprint']==saved['dataset']['motion_fingerprint']
    train=set(saved['dataset']['group_split']['train']);heldout=set(saved['dataset']['group_split']['validation'])
    assert {x['group_id'] for x in r['rows']}==train and len(train)==39 and len(heldout)==10 and not train&heldout
    for row in r['rows']:
        if args.branch=='student':
            row['rejection_reasons']=[x for x in row['rejection_reasons'] if x not in ('teacher_fault','no_supported_takeover')]
            row['qualified']=not row['rejection_reasons']
    if args.branch=='student':
        assert not arrays['teacher_selected'].any() and not arrays['label_valid'].any()
        np.testing.assert_array_equal(arrays['applied_action'],arrays['student_action'])
        r.update(correction_schema='ct_bc_diagnostic_queries_v1',teacher_queries_only=True,independent_student_control=True,
            valid_correction_frames=0,qualified=sum(x['qualified'] for x in r['rows']))
    else:
        np.testing.assert_array_equal(arrays['applied_action'],np.where(arrays['teacher_selected'][...,None],arrays['label'],arrays['student_action']))
    for key,value in before.items():torch.testing.assert_close(value,actor.state_dict()[key],atol=0,rtol=0)
    for name,sha in immutable.items():assert digest(name)==sha
    sources={}
    for source in (Path(__file__),ROOT/'scripts/agent/check_bc_local_feedback.py'):
        name=str(source.relative_to(ROOT));target=args.output/'snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);sources[name]=digest(source)
    r['source_sha256'].update(sources)
    r.update(branch=args.branch,trace_sha256=digest(path),probes_sha256=digest(args.output/'probes.npz'),
        anchor_times_s=list(ANCHORS),perturbations=[dict(variable=k,amplitude=v) for k,v in PERTURBATIONS],
        max_cloned_teacher_vs_live_action_difference=max(branch_errors),models_unchanged=True,
        checkpoint_total_actor_updates=saved['total_actor_updates'],immutable_sha256=immutable,
        pose_dataset_sha256=digest(args.pose_dataset),optimizer_updates=0,
        diagnostic_query_labels_exported_for_training=False,
        scope='Train39 poses x4 delays; physical dynamics unchanged. Student branch never applies teacher. Teacher branch actually takes over at10s without resetting shared position. Local probes are nominal requests, not delayed physical perturbations.')
    result_path.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('branch','num_envs','qualified','valid_correction_frames','max_cloned_teacher_vs_live_action_difference')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','pose-dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--branch',choices=('student','teacher'),required=True);run(p.parse_args())
