"""Check simulation-only delay selection and actual PD/sensor paths without physics."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config,from_sim2sim_config,from_record
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import make_env_cfg,sample_substep
from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT
from ct_mjlab.fudan_ppo_walk.selected_policy import selection_record,policy_inputs
from .check_disturbances import control_fixture
from . import evaluate


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);checks=[]
    def verify(name,condition):
        assert condition,name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('No real physics, network forward or optimizer updates')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as directory:
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','mjlab.envs.ManagerBasedRlEnv.step','torch.nn.Linear.forward','torch.optim.Adam.step','torch.cuda._lazy_init'):
            stack.enter_context(patch(name,side_effect=forbidden))
        cfg=from_sim2sim_config();training=from_task_config()
        verify('sim2sim_fixed4ms_both',cfg.action_delay and cfg.sensor_delay and cfg.action_delay_ms==(4.,4.) and cfg.sensor_delay_ms==(4.,4.))
        verify('training_delays_fixed2ms',training.action_delay and training.sensor_delay and training.action_delay_ms==(2.,2.) and training.sensor_delay_ms==(2.,2.))
        training_state=control_fixture(training).action.disturbances
        verify('training_runtime_delays_both_one_tick',training_state.action_steps.eq(1).all() and training_state.sensor_steps.eq(1).all())
        verify('pushes_still_off',not cfg.pushes and not training.pushes)
        verify('legacy_checkpoint_conditions_not_reinterpreted',not from_record().action_delay and not from_record().sensor_delay)
        for action,sensor in ((False,False),(True,False),(False,True),(True,True)):
            task=FudanWalkCfg()
            with patch.object(task.sim2sim,'randomize_action_delay',action),patch.object(task.sim2sim,'randomize_sensor_delay',sensor):
                sim=from_sim2sim_config(task)
                verify(f'independent_switches_{action}_{sensor}',sim.action_delay==action and sim.sensor_delay==sensor)
                verify(f'training_unchanged_{action}_{sensor}',from_task_config(task)==training)
        task=FudanWalkCfg()
        with patch.object(task.domain_rand,'delay_ms_range',[10.,10.]):
            verify('sim2sim_does_not_inherit_training_delay',from_sim2sim_config(task)==cfg)
        verify('physical_config_consumes_explicit_sim2sim_delays',make_env_cfg(disturbances=cfg).actions['pd'].disturbances==cfg)
        verify('training_physical_config_uses_training_only',make_env_cfg().actions['pd'].disturbances==training)
        env=control_fixture(cfg);state=env.action.disturbances
        verify('runtime_delay_steps_both2',state.action_steps.eq(2).all() and state.sensor_steps.eq(2).all())
        env.action.process_actions(torch.full((3,6),.1));torques=[]
        for tick in range(5):
            env.action.apply_actions();torques.append(env.action.torques[:,0].clone())
        observed=torch.stack(torques)
        expected=torch.tensor([0.,0.,3.,3.,3.])[:,None].expand(-1,3)
        verify('new_action_reaches_PD_after_two2ms_ticks',torch.allclose(observed,expected,atol=1e-4))
        for tick in range(1,6):
            theta=torch.tensor(tick*.01)
            env.env.sim.data.qpos[:,3:7]=torch.tensor([torch.cos(theta/2),0.,torch.sin(theta/2),0.])
            env.env.sim.data.qvel[:,3]=tick*4.
            sample_substep(env.env)
        env._update_state();env.commands[:]=torch.tensor([.7,.2,.26]);env.action._raw_actions.fill_(.3)
        obs,critic=env.observations();clean=env.frame()
        verify('gyro_delivered_from_two_ticks_ago',obs[:,0].eq(3.).all())
        verify('gravity_delivered_from_two_ticks_ago',torch.allclose(obs[:,3],torch.full((3,),torch.sin(torch.tensor(.03))),atol=1e-7))
        verify('commands_and_previous_action_are_current',torch.equal(obs[:,6:9],clean[:,6:9]) and torch.equal(obs[:,19:],clean[:,19:]))
        verify('critic_keeps_current_state',torch.equal(critic[:,3:28],clean))
        biased,_=policy_inputs(obs,obs.repeat(1,5),3.,torch.zeros(3),torch.full((3,),2*torch.pi/180.))
        verify('pitch2_reference_rotates_delivered_gravity',torch.allclose(biased[:,3],torch.full((3,),torch.sin(torch.tensor(.03-2*torch.pi/180.))),atol=1e-7))
        other=state.sensors[:,[0,2]].clone();env.action.reset(torch.tensor([1]))
        verify('reset_keeps_other_sensor_queues',torch.equal(other,state.sensors[:,[0,2]]))
        verify('reset_fills_fresh_sensor_values',torch.equal(state.sensors[:,1],env.action.sensor_frame()[1].expand(len(state.sensors),-1)))
        verify('reset_keeps_fixed_delays',state.action_steps.eq(2).all() and state.sensor_steps.eq(2).all())
        selected=selection_record();verify('selected8000_pitch2_integral_off',selected['parking_pitch_deg']==2 and not selected['integral_enabled'])
        class BeforePhysics(Exception):pass
        for selected_recipe in (None,selected):
            output=Path(directory)/('selected' if selected_recipe else 'explicit');received=[]
            def factory(*a,**kw):
                received.append(kw['disturbances']);raise BeforePhysics()
            arguments=SimpleNamespace(checkpoint=Path(selected['checkpoint']),selection=selected_recipe,diagnostic=True,
                diagnostic_height=None,output=output,batch_size=512,deadline=float('inf'))
            metadata=dict(reset_contract=RESET_V2_CONTRACT,disturbances=training.record())
            with patch.object(evaluate,'load_policy',return_value=(None,metadata)),patch('ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk.FudanEnv',side_effect=factory):
                try:evaluate.gpu(arguments)
                except BeforePhysics:pass
            mode='selected' if selected_recipe else 'explicit'
            protocol=json.loads((output/'protocol.json').read_text())
            verify(mode+'_GPU_entry_uses_sim2sim_delays',received==[cfg])
            verify(mode+'_records_actual_and_training_separately',protocol['disturbances']==cfg.record() and protocol['training_disturbances']==training.record())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(passed=True,count=len(checks),checks=checks,sim2sim=cfg.record(),training=training.record(),
        physics_steps=0,model_forwards=0,optimizer_updates=0,scope='Real PD/sensor methods on synthetic states and GPU entry before physical construction; not behavior validation')
    with args.output.open('x') as output:output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
