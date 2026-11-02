"""Exercise real delay/PD/sensor/push paths with synthetic states, without physics or learning."""
import argparse
from contextlib import ExitStack,redirect_stdout
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from ct_mjlab.fudan_ppo_walk.disturbances import DisturbanceConfig,DisturbanceState,from_task_config,from_record
from ct_mjlab.fudan_ppo_walk.config import FudanWalkCfg
from ct_mjlab.fudan_ppo_walk.env import PhysicalAction,PhysicalActionCfg,sample_substep,make_env_cfg
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT
from ct_mjlab.fudan_ppo_walk.measurement import configure_samples
from ct_mjlab.fudan_ppo_walk import __main__ as cli
from ct_mjlab.fudan_ppo_walk.runner import FudanRunner
from .reset_v2_fixture import fixture


def control_fixture(cfg,n=3,seed=43):
    env=fixture(n,RESET_CONTRACT);physics=env.env;robot=env.robot
    old=env.action
    physics.cfg=SimpleNamespace(seed=seed);physics.physics_dt=.002;physics.step_dt=.01
    physics.sim.data.ctrl=torch.zeros(n,physics.sim.mj_model.nu)
    physics.sim.data.xfrc_applied=torch.zeros(n,physics.sim.mj_model.nbody,6)
    robot.data.default_joint_pos=old.reset_joint_positions.clone()
    robot.indexing.joint_v_adr=torch.tensor([int(physics.sim.mj_model.joint(name).dofadr[0]) for name in robot.joint_names])
    def find(names,**kw):return [robot.joint_names.index(name) for name in names],list(names)
    robot.find_joints=find
    robot.set_joint_effort_target=lambda value,**kw:setattr(physics,'efforts',value.clone())
    action=PhysicalAction(PhysicalActionCfg(entity_name='robot',reference_randomization=False,
        motor_randomization=False,randomize_delays=False,action_delay_s=0.,disturbances=cfg),physics)
    env.action=action;physics.action_manager.get_term=lambda name:action
    env.disturbance_config=cfg;env.update_push_scale()
    action.reset();action.disturbances.sensors[:]=action.sensor_frame()
    env.randomized=False
    return env


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(11)
    checks=[]
    def verify(name,value):
        assert bool(value),name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('Real physics/learning forbidden')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as directory:
        directory=Path(directory)
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2',
                'mjlab.envs.ManagerBasedRlEnv.step','torch.optim.Adam.step','torch.nn.Linear.forward'):
            stack.enter_context(patch(name,side_effect=forbidden))
        default=DisturbanceConfig();rng=torch.get_rng_state().clone()
        state=DisturbanceState(default,4,'cpu')
        verify('all_off_does_not_consume_rng',torch.equal(rng,torch.get_rng_state()))
        verify('old_metadata_means_all_off',from_record()==default)
        verify('registered_environment_without_seed_supported',not control_fixture(default,seed=None).disturbance_config.pushes)
        for flags in itertools.product((False,True),repeat=3):
            cfg=DisturbanceConfig(*flags,sensor_delay_ms=(2.,6.))
            verify('config_round_trip_'+str(flags),from_record(json.loads(json.dumps(cfg.record())))==cfg)
        for values in ({'sensor_delay':True},{'action_delay_ms':(-1.,2.)},{'sensor_delay_ms':(4.,2.)},
                {'action_delay_ms':(0.,float('nan'))},{'push_interval_s':0.},{'max_push_vel':float('inf')}):
            try:DisturbanceConfig(**values)
            except ValueError:rejected=True
            else:rejected=False
            verify('invalid_'+str(values),rejected)
        domain=FudanWalkCfg.domain_rand
        for flags in itertools.product((False,True),repeat=3):
            with patch.object(domain,'randomize_action_delay',flags[0]),patch.object(domain,'randomize_sensor_delay',flags[1]),\
                    patch.object(domain,'push_robots',flags[2]),patch.object(domain,'sensor_delay_ms_range',[2.,8.]):
                cfg=from_task_config()
                verify('code_switches_independent_'+str(flags),(cfg.action_delay,cfg.sensor_delay,cfg.pushes)==flags)
                verify('environment_reads_code_config_'+str(flags),make_env_cfg().actions['pd'].disturbances==cfg)
                verify('old_metadata_not_reinterpreted_'+str(flags),from_record()==default)
        # 不经策略推理，直接检查真实PD收到的新动作在哪个2ms拍出现。
        for delay in (0.,2.,4.,6.,8.,10.):
            env=control_fixture(DisturbanceConfig(action_delay=True,action_delay_ms=(delay,delay)))
            action=env.action;action.process_actions(torch.full((3,6),.1));observed=[]
            for tick in range(8):
                action.apply_actions();observed.append(action.torques[:,0].clone())
            actual=torch.stack(observed)
            expected=torch.tensor([0. if tick<int(delay/2) else 3. for tick in range(8)])[:,None].expand(-1,3)
            verify(f'action_{delay}ms_reaches_actual_pd',torch.allclose(actual,expected,atol=1e-4))
            verify(f'action_{delay}ms_keeps_raw_observation',action.raw_action.eq(.1).all())
        a=control_fixture(default);b=control_fixture(DisturbanceConfig(action_delay=True,action_delay_ms=(0.,0.)))
        for tick in range(9):
            value=torch.full((3,6),tick*.03)
            for env in (a,b):env.action.process_actions(value);env.action.apply_actions()
            verify(f'off_matches_zero_delay_pd_{tick}',torch.equal(a.action.torques,b.action.torques))

        env=control_fixture(DisturbanceConfig(sensor_delay=True,sensor_delay_ms=(4.,4.)))
        state=env.action.disturbances
        for tick in range(1,6):
            env.env.sim.data.qvel[:,3]=tick*4.
            sample_substep(env.env)
        env._update_state();env.commands[:]=torch.tensor([.7,.2,.26]);env.action._raw_actions.fill_(.3)
        obs,critic=env.observations();clean=env.frame()
        verify('sensor_4ms_uses_two_physics_ticks_ago',obs[:,0].eq(3.).all() and clean[:,0].eq(5.).all())
        verify('commands_and_previous_action_not_delayed',torch.equal(obs[:,6:9],clean[:,6:9]) and torch.equal(obs[:,19:],clean[:,19:]))
        verify('critic_keeps_current_state',torch.equal(critic[:,3:28],clean))
        env.history=obs.repeat(1,5)
        verify('history_contains_delivered_sensor_frame',torch.equal(env.history[:,-25:],obs))
        before=state.sensors[:,[0,2]].clone();ids=torch.tensor([1])
        env.action.reset(ids)
        verify('local_reset_keeps_other_sensor_queues',torch.equal(before,state.sensors[:,[0,2]]))
        verify('local_reset_has_fresh_sensor_fill',torch.equal(state.sensors[:,1],env.action.sensor_frame()[1].expand(len(state.sensors),-1)))

        cfg=DisturbanceConfig(action_delay=True,action_delay_ms=(10.,10.),pushes=True,push_interval_s=.006,max_push_vel=.1)
        env=control_fixture(cfg);action=env.action;state=action.disturbances
        action.process_actions(torch.full((3,6),.2))
        blocks=[]
        for tick in range(7):
            action.apply_actions();blocks.append(env.env.sim.data.xfrc_applied[:,action._base_id].clone())
        blocks=torch.stack(blocks)
        verify('push_only_on_ticks_3_and_6',blocks.abs().sum(dim=(1,2)).ne(0).tolist()==[False,False,True,False,False,True,False])
        verify('push_no_external_torque',blocks[:,:,3:].eq(0).all())
        verify('push_counts_are_actual_pulses',state.push_count.tolist()==[2,2,2])
        verify('push_vertical_half_scale',torch.all(blocks[:,:,:2].abs()<=action.push_force_scale[None,:,None])
            and torch.all(blocks[:,:,2].abs()<=action.push_force_scale[None,:]*.5))
        unchanged=state.actions[:,[0,2]].clone();seeds=state.push_seeds.copy();ticks=state.ticks.clone()
        action.reset(torch.tensor([1]))
        verify('local_reset_clears_only_selected_actions',state.actions[:,1].eq(0).all() and torch.equal(unchanged,state.actions[:,[0,2]]))
        verify('local_reset_push_timer_and_seed',state.ticks[1]==0 and torch.equal(state.ticks[[0,2]],ticks[[0,2]])
            and state.push_seeds[1]!=seeds[1] and np.array_equal(state.push_seeds[[0,2]],seeds[[0,2]]))

        # 核对GPU评估夹具保存的条件；不构造CPU物理后端。
        cfg=DisturbanceConfig(action_delay=True,action_delay_ms=(4.,4.),sensor_delay=True,sensor_delay_ms=(4.,4.),
            pushes=True,push_interval_s=.006,max_push_vel=.1)
        env=control_fixture(cfg,2)
        items=[dict(id=f'fixture_{i}',seed=42+i,randomized=False,
            case=dict(id='switch_check',segments=[dict(start=0.,command=[.5,0.,.26])])) for i in range(2)]
        original_save=mujoco.mj_saveModel
        with patch('mujoco.mj_saveModel',side_effect=lambda model,path:original_save(model.model,path)):
            obs,_=configure_samples(env,items,directory/'fixtures')
        mapping=json.loads((directory/'fixtures/mapping.json').read_text())
        verify('gpu_fixture_saves_disturbance_config',mapping['disturbances']==cfg.record())
        with np.load(directory/'fixtures/fixture_0_initial.npz') as initial:
            verify('gpu_fixture_saves_action_delay_steps',int(initial['action_delay_steps'])==2)
            verify('gpu_fixture_saves_sensor_delay_steps',int(initial['sensor_delay_steps'])==2)
            verify('gpu_fixture_saves_push_seed',int(initial['push_seed'])==int(env.action.disturbances.push_seeds[0]))

        # 入口读取代码配置，并拒绝不同条件的预检；配置变更记录到恢复报告。
        proof=directory/'proof.json';parent=directory/'parent.pt'
        proof.write_text(json.dumps(dict(passed=True,reset_contract=RESET_CONTRACT,kind='physical-visible-precheck',num_envs=20,physics_steps=1000)))
        torch.save({},parent)
        argv=['train','--num-envs','4','--iterations','1','--precheck',str(proof),'--log-dir',str(directory/'run')]
        with patch.object(cli,'FudanEnv',side_effect=forbidden),patch.object(domain,'randomize_action_delay',True):
            try:cli.main(argv)
            except ValueError as exc:rejected='precheck' in str(exc)
            else:rejected=False
        verify('different_precheck_rejected_before_environment',rejected)
        proof_data=json.loads(proof.read_text());proof_data['disturbances']=DisturbanceConfig(action_delay=True).record()
        proof.write_text(json.dumps(proof_data));entered={}
        class EntryRunner:
            iteration=0
            def __init__(self,env,**kw):pass
            def load(self,path,**kw):entered.update(kw);return {}
            def learn_until(self,*a,**kw):return {}
        with patch.object(cli,'FudanEnv',return_value=SimpleNamespace(close=lambda:None)),patch.object(cli,'FudanRunner',EntryRunner),patch.object(cli,'snapshot'),patch.object(domain,'randomize_action_delay',True),redirect_stdout(io.StringIO()):
            cli.main(argv+['--resume',str(parent)])
        verify('code_config_resume_condition_change_recorded',entered=={'allow_disturbance_change':True})

        # 实际保存/恢复路径，网络只读写state_dict，绝不前向或优化。
        learner=object.__new__(FudanRunner);learner.device='cpu';learner.log_dir=directory
        learner.env=SimpleNamespace(disturbance_config=cfg,reset_contract=RESET_CONTRACT,
            num_envs=2,total_steps=49,speed_limit=2.,env=SimpleNamespace(common_step_counter=49))
        learner.policy=torch.nn.Linear(1,1);encoder=torch.nn.Linear(1,1)
        learner.alg=SimpleNamespace(optimizer=torch.optim.Adam(learner.policy.parameters(),lr=.001),
            extra_optimizer=torch.optim.Adam(encoder.parameters(),lr=.001),learning_rate=.001)
        learner.iteration=1;learner.transitions=96;learner.primary_updates=learner.encoder_updates=20
        learner.partial_update=False;learner.initialization='random_from_scratch_seed_1';learner.resume_from=None
        saved=learner.save('condition-check.pt')
        verify('checkpoint_saves_full_config',torch.load(saved,weights_only=False)['disturbances']==cfg.record())
        restored=learner.load(saved)
        verify('checkpoint_restores_condition_record',restored['disturbances']==cfg.record())
        learner.env.disturbance_config=default
        try:learner.load(saved)
        except ValueError:rejected=True
        else:rejected=False
        verify('silent_resume_condition_change_rejected',rejected)
        restored=learner.load(saved,allow_disturbance_change=True)
        verify('explicit_resume_reports_both_conditions',restored['source_disturbances']==cfg.record() and restored['disturbances']==default.record())
        legacy=torch.load(saved,weights_only=False);legacy.pop('disturbances');torch.save(legacy,directory/'legacy.pt')
        restored=learner.load(directory/'legacy.pt')
        verify('legacy_checkpoint_loads_all_off',restored['disturbances']==default.record())

        devices=['cpu']
        if torch.cuda.is_available():
            devices.append('cuda:0')
            cpu=DisturbanceState(cfg,2,'cpu');gpu=DisturbanceState(cfg,2,'cuda:0')
            cpu.reset(slice(None),torch.zeros(2,16));gpu.reset(slice(None),torch.zeros(2,16,device='cuda:0'))
            for tick in range(12):
                value=torch.full((2,6),tick*.1);sensors=torch.full((2,16),float(tick))
                verify(f'cuda_action_queue_{tick}',torch.equal(cpu.action(value),gpu.action(value.cuda()).cpu()))
                cpu.sample(sensors);gpu.sample(sensors.cuda())
                verify(f'cuda_sensor_queue_{tick}',torch.equal(cpu.observed(),gpu.observed().cpu()))
                quat=torch.tensor([[1.,0.,0.,0.],[.70710678,0.,0.,.70710678]])
                scale=torch.tensor([20.,30.])
                verify(f'cuda_force_schedule_{tick}',torch.allclose(cpu.force(quat,scale),gpu.force(quat.cuda(),scale.cuda()).cpu(),atol=2e-5))
    result=dict(passed=True,checks=checks,count=len(checks),physics_steps=0,model_forwards=0,optimizer_updates=0,
        tensor_devices=devices,
        scope='Actual task PD, queues, force schedule, GPU fixture and CLI with synthetic states; no dynamics or behavior claim')
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
