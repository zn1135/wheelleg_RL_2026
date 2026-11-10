"""Nonphysical checks of integral state, real v2 command slots and original history advancement."""
import argparse,json,hashlib
import numpy as np
from pathlib import Path
from types import MethodType
from contextlib import ExitStack
from unittest.mock import patch
import torch
from ct_mjlab.envs.fudan_ppo_walk.parking_integral import ParkingIntegral,feedback_velocity,REASONS,configuration
from ct_mjlab.fudan_ppo_walk.measurement import FIELDS,update_evaluation_commands
from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT
from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig,DisturbanceState
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import PhysicalAction
from .reset_v2_fixture import fixture
from .diagnose_parking_integral import restore_initial,compare_initial,policy_tick,BRANCHES,HEIGHTS


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    checks=[]
    def verify(name,condition):
        assert condition,name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('Physics/model/optimizer forbidden before budget')
    raw=torch.tensor([[0.,0.,.3]]*3);angles=torch.zeros(3,2);contact=torch.tensor([[[20.,20.,0.]]*3]*10)
    def ready(enabled=True):
        c=ParkingIntegral(3,enabled=[enabled]*3);c.observe_contacts(contact);return c
    def cycle(c,k,feedback,cmd=raw,pose=angles):
        actual,_=c.begin(k,cmd,pose);c.end(cmd,feedback,pose);return actual
    with ExitStack() as stack:
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','torch.nn.Linear.forward','torch.optim.Adam.step','torch.cuda._lazy_init','mjlab.envs.ManagerBasedRlEnv.step'):
            stack.enter_context(patch(name,side_effect=forbidden))
        verify('default_disabled',not configuration()['enabled'])
        c=ready();out=[]
        for k in range(51):out.append(cycle(c,k,torch.full((3,),-.05)))
        verify('gate_waits_50_complete_intervals',all(torch.equal(x,raw) for x in out) and c.active.all())
        verify('first_feedback_after_gate_and_after_action',torch.allclose(c.integral,torch.full((3,),.0005)))
        actual=cycle(c,51,torch.full((3,),-.05))
        verify('negative_velocity_positive_compensation',torch.all(actual[:,0]>0))
        verify('encoder_scale_two_restored',torch.equal(feedback_velocity(torch.tensor([-.1]),True),torch.tensor([-.05])))
        for k in range(52,452):cycle(c,k,torch.full((3,),-10.))
        verify('integral_and_speed_clamped',torch.allclose(c.integral,torch.full((3,),.1)) and torch.allclose(c.compensation,torch.full((3,),.1)))
        cycle(c,452,torch.full((3,),.5));verify('reverse_error_unwinds_saturation',torch.all(c.integral<.1))
        c.reset(torch.tensor([1]));verify('reset_is_local',c.integral[0]>0 and c.integral[1]==0 and c.integral[2]>0 and c.contact_count.tolist()==[10,0,10])
        for name,column in [('moving',0),('turning',1)]:
            d=ready();d.integral[:]=.05;command=raw.clone();command[0,column]=.1
            cycle(d,0,torch.zeros(3),command)
            verify(name+'_clears_only_matching_env',d.integral.tolist()==[0.,.05000000074505806,.05000000074505806] and d.reason[0]==REASONS[name])
        for name,pose,force in [('unstable',torch.tensor([[10.,0.]]*3),contact),('unsupported',angles,torch.tensor([[[1.,20.,0.]]*3]*10)),('body_contact',angles,torch.tensor([[[20.,20.,1.01]]*3]*10))]:
            d=ready();d.observe_contacts(force);d.integral[:]=.04;cycle(d,0,torch.zeros(3),pose=pose)
            verify(name+'_clears',not d.integral.any() and (d.reason==REASONS[name]).all())
        d=ParkingIntegral(3,enabled=[True]*3);d.observe_contacts(contact[:9]);cycle(d,0,torch.zeros(3))
        verify('requires_10_real_contact_samples',(d.reason==REASONS['contact_warmup']).all())
        d.observe_contacts(contact[:1]);cycle(d,1,torch.zeros(3))
        verify('exact20ms_arithmetic_mean',torch.equal(d.contacts.mean(0),contact.mean(0)) and (d.reason==REASONS['waiting']).all())
        sample=contact[:1].clone();sample[:,:,0]=120.;d.observe_contacts(sample)
        verify('rolling20ms_replaces_oldest',torch.equal(d.contacts.mean(0)[:,0],torch.full((3,),30.)))
        d=ready();d.begin(0,raw,angles)
        try:d.begin(1,raw,angles);raised=False
        except RuntimeError:raised=True
        verify('duplicate_begin_rejected',raised)
        d.end(raw,torch.zeros(3),angles)
        try:d.end(raw,torch.zeros(3),angles);raised=False
        except RuntimeError:raised=True
        verify('duplicate_update_rejected',raised)
        for field in ('command','angle','contact','feedback'):
            d=ready();bad=torch.full((3,),float('nan'))
            try:
                if field=='contact':d.observe_contacts(contact*float('nan'))
                else:
                    d.begin(0,raw*float('nan') if field=='command' else raw,angles*float('inf') if field=='angle' else angles)
                    if field=='feedback':d.end(raw,bad,angles)
                raised=False
            except FloatingPointError:raised=True
            verify('nonfinite_'+field+'_terminates',raised)
        d=ready();d.integral[:]=.05;d.ticks[:]=50;d.last_good[:]=True
        actual,_=d.begin(0,raw,angles);d.end(raw,torch.zeros(3),angles)
        verify('error_uses_raw_not_compensated_command',torch.allclose(d.integral,torch.full((3,),.05)))
        d.begin(1,raw,angles);d.end(raw,torch.zeros(3),torch.full_like(angles,11.))
        verify('post_step_instability_clears_immediately',not d.integral.any() and not d.compensation.any() and not d.active.any())
        d=ready()
        for k in range(40):cycle(d,k,torch.zeros(3))
        cycle(d,40,torch.zeros(3),pose=torch.full_like(angles,11.))
        for k in range(41,91):cycle(d,k,torch.zeros(3))
        verify('interrupted_gate_requires_full_new_half_second',not d.active.any())
        cycle(d,91,torch.zeros(3));verify('reactivates_after_50_new_intervals',d.active.all())
        d=ready();d.integral[0]=float('inf')
        try:d.begin(0,raw,angles);raised=False
        except FloatingPointError:raised=True
        verify('nonfinite_integral_terminates',raised)
        d=ready();force=contact.clone();force[:,:,2]=1.;d.observe_contacts(force);cycle(d,0,torch.zeros(3))
        verify('body_contact_equal_one_allowed',(d.reason==REASONS['waiting']).all())
        torch.manual_seed(931);env=fixture(3,reset_contract=RESET_V2_CONTRACT);obs,_,history=env.reset()
        disabled=ParkingIntegral(3);before=env.history.clone();command=raw.clone()
        # A deterministic algebraic action probe, not a neural model or trained checkpoint.
        probe=lambda o,h:o[:,:6]+h[:,-6:]
        expected=update_evaluation_commands(env,command,obs.clone());expected_h=env.history.clone();expected_action=probe(expected,expected_h)
        actual,_=disabled.begin(0,command,angles);actual_obs=update_evaluation_commands(env,actual,obs.clone())
        verify('disabled_obs_history_action_identical',torch.equal(expected,actual_obs) and torch.equal(env.history,expected_h) and torch.equal(probe(actual_obs,env.history),expected_action))
        verify('old_history_frames_never_rewritten',torch.equal(env.history[:,:100],before[:,:100]))
        obs,_,_,_,_,h=env.step(expected_action,auto_reset=False,resample=False);disabled.end(command,torch.zeros(3),angles)
        verify('original_step_advances_history_once',torch.equal(h[:,:100],expected_h[:,25:]) and env.mock_physics_calls==2 and disabled.updates==1)
        enabled=ready();enabled.integral[:]=.05;enabled.ticks[:]=50;enabled.last_good[:]=True
        old=env.history.clone();cmd,_=enabled.begin(0,raw,angles);o=update_evaluation_commands(env,cmd,obs)
        verify('enabled_only_current_command_and_last_frame',torch.equal(env.history[:,:100],old[:,:100]) and torch.equal(o[:,6],torch.full((3,),.1)) and torch.equal(env.history[:,-19],o[:,6]))
        # Exercise the real archive restore against all 135 records, using tensors only.
        initial_checks=[];root=Path('artifacts/fudan_ppo_walk/fudan-gpu-replay-8000-20261102')
        for height in HEIGHTS:
            source=root/height/'gpu15';items=json.loads((source/'protocol.json').read_text())['samples']
            f=fixture(45,reset_contract=RESET_V2_CONTRACT);action=f.action
            # The lightweight robot fixture lacks the GPU scene's terrain body/geom; mirror archive tensor shapes.
            with np.load(source/'fixtures'/(items[0]['id']+'_initial.npz')) as archive:
                for key in FIELDS:setattr(f.env.sim.model,key,torch.full((45,*archive[key].shape),99.,dtype=torch.float32))
            f.disturbance_config=DisturbanceConfig();action.num_envs=45;action._env=f.env
            action.sensor_frame=MethodType(PhysicalAction.sensor_frame,action)
            action.disturbances=DisturbanceState(f.disturbance_config,45,'cpu');action.push_force_scale=torch.ones(45)
            action.substep_rows=[];action._torques=action.torques
            action._motor_response_torque=torch.ones(45,6);action._action_history=torch.ones(2,45,6)
            action._spring_force_bias=torch.ones(45,2);action.action_delay_steps=torch.ones(45,dtype=torch.long)
            action.wheel_action_delay_steps=torch.ones(45,dtype=torch.long)
            f.env.sim.data.ctrl=torch.ones(45,8);f.env.sim.data.qacc_warmstart=torch.ones_like(f.env.sim.data.qvel)
            # Contaminate both nominal and pose so a no-op restore cannot pass.
            action._nominal_pos.add_(.02);f.env.sim.data.qpos[:,7:]+=.01;f.history.fill_(99);f.previous.fill_(8)
            folders={b:a.output.parent/'fixture-checks'/height/b for b in BRANCHES}
            for folder in folders.values():(folder/'fixtures').mkdir(parents=True,exist_ok=False)
            with patch.object(action,'rebuild_reset_poses',side_effect=forbidden):
                restored,_=restore_initial(f,items,source/'fixtures',folders,initial_checks,height)
            verify(height+'_restore_uses_no_physics_steps',f.mock_physics_calls==0)
            verify(height+'_restore_clears_actions_and_solver_cache',not action._motor_response_torque.any() and not action._action_history.any()
                and not f.env.sim.data.qacc_warmstart.any() and not f.env.sim.data.ctrl.any() and not f.previous.any())
            verify(height+'_fresh_state_and_v2_history',not f.linear.any() and torch.equal(f.previous_position,f.robot.data.root_link_pos_w)
                and torch.equal(f.history,restored.repeat(1,5)) and not f._history_needs_fill.any())
            verify(height+'_all_branches_keep_original_fixture_bytes',all((folder/'fixtures'/p.name).read_bytes()==p.read_bytes()
                for folder in folders.values() for p in (source/'fixtures').iterdir() if p.is_file()))
            state=action.disturbances
            verify(height+'_restore_clears_disturbances',not state.actions.any() and not state.ticks.any() and not action.push_force_scale.any()
                and torch.equal(state.sensors[0],action.sensor_frame()))
        verify('all_135_archived_states_identical_on_nonphysical_backend',len(initial_checks)==135 and all(r['all_arrays_identical'] for r in initial_checks))
        (a.output.parent/'fixture-initial-consistency.json').write_text(json.dumps(initial_checks,indent=2)+'\n')
        target=np.array([0.,.1,-.2],dtype=np.float32);different=target.copy();different[1]=np.nextafter(target[1],np.float32(1.))
        evidence=compare_initial({'qpos':target},{'qpos':different},a.output.parent/'difference-check','one_ulp')
        verify('one_ulp_difference_is_rejected_without_tolerance',not evidence['all_arrays_identical'] and evidence['differences']['qpos']['indices']==[[1]])
        with np.load(evidence['differences']['qpos']['arrays']) as saved:
            verify('difference_keeps_both_arrays_and_every_delta',np.array_equal(saved['actual'],different) and np.array_equal(saved['expected'],target)
                and np.array_equal(saved['difference'],different.astype(float)-target.astype(float)))
        verify('difference_json_is_saved',(a.output.parent/'difference-check/initial-differences/one_ulp.json').is_file())
        from .report_parking_integral import missing_report
        missing=a.output.parent/'missing-report-check';missing.mkdir(parents=True,exist_ok=False)
        (missing/'initial-consistency.json').write_text(json.dumps([dict(all_arrays_identical=True),dict(all_arrays_identical=False,differences=evidence['differences'])]))
        (missing/'initialization-result.json').write_text(json.dumps(dict(passed=False,error='intentional_one_ulp_for_report_check')))
        difference_folder=missing/'h022/truth_integral/initial-differences';difference_folder.mkdir(parents=True)
        for path in (a.output.parent/'difference-check/initial-differences').iterdir():
            (difference_folder/path.name).write_bytes(path.read_bytes())
        missing_report(missing,root)
        report=json.loads((missing/'comparison.json').read_text())
        verify('missing_report_uses_actual_error_without_completed_pipeline',report['initial_check_failure']=='intentional_one_ulp_for_report_check'
            and report['original_model_source_input_unchanged'] is None)
        verify('missing_report_counts_failed_and_unexamined_separately',report['initial_checks_passed']==1 and report['initial_checks_failed']==1
            and report['initial_checks_unexamined']==133 and len(report['inventory'])==135)
        verify('missing_report_does_not_claim_behavior_or_video_acceptance',report['complete_trajectories']==0
            and report['final_acceptance']=='not_ready_no_dynamic_results' and all(x is None for x in report['parking_passed'].values()))
        verify('missing_report_links_saved_difference_arrays',report['exact_failed_values_available'] and len(report['initial_difference_files'])==2)
        # Deliberately make before/after feedback differ; the production tick must choose after.
        f=fixture(3,reset_contract=RESET_V2_CONTRACT);f.randomized=False;obs,_,_=f.reset()
        f.env.sim.data.qvel.zero_();f.env.sim.data.qvel[:,0]=-.04;f.previous_position.copy_(f.position);f._update_state()
        f.action.substep_rows=[];original_step=f.env.step
        def sample_row(_):
            row=torch.zeros(3,18);row[:,:3]=f.position;row[:,14:16]=20.;return row
        def step_and_sample(actions):
            original_step(actions);f.action.substep_rows[:]=[sample_row(f.env)]*5
        f.env.step=step_and_sample
        class ProbePolicy:
            def __init__(self):self.actors=0;self.encoders=0;self.sample_steps=[]
            def encode(self,history):
                self.encoders+=1;self.sample_steps.append(f.mock_physics_calls);return history[:,-6:-3].clone()
            def act_inference(self,obs,history):
                self.actors+=1;self.actor_history=history.clone();estimate=self.encode(history)
                return torch.full((3,6),-.1),estimate
        probe=ProbePolicy();c=ParkingIntegral(3,enabled=[False,True,True]);c.observe_contacts(contact);c.ticks[:]=50;c.last_good[:]=True
        calls=f.mock_physics_calls
        with patch('scripts.agent.fudan_ppo_walk.diagnose_parking_integral.gpu_row',side_effect=sample_row):
            obs,_,_,values=policy_tick(f,probe,c,0,raw,obs)
            verify('truth_feedback_is_current_step_endpoint',torch.equal(values['feedback_m_s'][1,0],f.linear[1,0])
                and values['truth_m_s'][1,0]==0 and c.integral[1]==-f.linear[1,0]*.01)
            verify('encoder_feedback_is_new_history_divided_by_two',values['encoder_raw'][2,0]==0
                and values['feedback_encoder_raw'][2,0]==-.1 and values['feedback_m_s'][2,0]==-.05 and c.integral[2]==.0005)
            verify('actor_once_feedback_encoder_after_step',probe.actors==1 and probe.encoders==2 and probe.sample_steps==[calls,calls+1])
            verify('feedback_probe_does_not_append_history',torch.equal(f.history[:,:100],probe.actor_history[:,25:]) and f.mock_physics_calls==calls+1 and c.updates==1)
            verify('new_feedback_not_used_in_same_tick',not values['compensation'].any())
            previous_integral=c.integral.clone();obs,_,_,next_values=policy_tick(f,probe,c,1,raw,obs)
            verify('endpoint_feedback_compensates_next_tick_only',torch.equal(next_values['compensation'][:,0],previous_integral) and c.updates==2 and probe.actors==2)
            verify('baseline_stays_identical_during_production_tick',torch.equal(next_values['policy_command'][0],raw[0]) and c.integral[0]==0)
            bad=ProbePolicy();bad.encode=lambda h:torch.full((3,3),float('nan'))
            try:policy_tick(f,bad,c,2,raw,obs);raised=False
            except FloatingPointError:raised=True
            verify('nonfinite_endpoint_encoder_stops_before_integral_commit',raised and c.updates==2)
    from .report_parking_integral import parking_metrics
    n=2501;trace=dict(time=np.arange(n)*.002,position=np.zeros((n,3)),linear_world=np.zeros((n,3)))
    trace['linear_world'][:,0]=np.where(np.arange(n)%2,.02,-.02)
    verify('score_speed_magnitude_no_signed_cancellation',abs(parking_metrics(trace,0,5)['mean_horizontal_speed_m_s']-.02)<1e-12 and not parking_metrics(trace,0,5)['passed'])
    trace['linear_world'][:]=0.;trace['position'][-1,0]=.051
    verify('score_offset_includes_window_endpoint',not parking_metrics(trace,0,5)['passed'])
    trace['position'][-1,0]=.05
    verify('score_inclusive_5cm_boundary',parking_metrics(trace,0,5)['passed'])
    from ct_mjlab.fudan_ppo_walk.source_migration import verify_migration
    verify_migration()
    protected=['fudan_ppo_walk/env.py','fudan_ppo_walk/network.py','fudan_ppo_walk/contract.py','fudan_ppo_walk/policy.py','envs/chuanliantui/pd_action.py','envs/chuanliantui/huananhu_action.py']
    source=Path('logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/source/ct_mjlab')
    for name in protected:
        verify('8000_training_source_unchanged:'+name,(source/name).read_bytes()==(Path('artifacts/fudan_ppo_walk/fudan-structure-alignment-20261103/before/mjlab_training/src/ct_mjlab')/name).read_bytes())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),Path('scripts/agent/fudan_ppo_walk/diagnose_parking_integral.py'),Path('scripts/agent/fudan_ppo_walk/report_parking_integral.py'),
        Path('scripts/agent/fudan_ppo_walk/run_parking_integral.py'),
        Path('scripts/agent/fudan_ppo_walk/reset_v2_fixture.py'),Path('mjlab_training/src/ct_mjlab/fudan_ppo_walk/measurement.py'),
        Path('mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/parking_integral.py'),Path('mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk_config.py')]
    a.output.write_text(json.dumps(dict(passed=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},checks=checks,count=len(checks),
        restored_fixture_checks=135,physics_steps=0,model_forwards=0,optimizer_updates=0,cuda_initializations=0,
        action_probe='deterministic algebra; original policy code remains protected; fixture restore/endpoint feedback use nonphysical backend'),indent=2)+'\n')
    print(json.dumps(dict(passed=True,count=len(checks))))

if __name__=='__main__':main()
