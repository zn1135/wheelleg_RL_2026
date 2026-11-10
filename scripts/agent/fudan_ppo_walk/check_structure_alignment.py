"""Compare the pre-migration implementation to production without physics or Adam updates."""
import argparse
import ast
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, MethodType
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs.fudan_ppo_walk import fudan_ppo_walk as current
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, FudanWalkCfgPPO, section_dict
from ct_mjlab.fudan_ppo_walk import contract
from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
from ct_mjlab.rsl_rl.algorithms.fudan_ppo import PPO
from ct_mjlab.envs import task_registry
from .reset_v2_fixture import fixture

ROOT=Path(__file__).resolve().parents[3]
ARTIFACT=ROOT/'artifacts/fudan_ppo_walk/fudan-structure-alignment-20261103'
CHECKPOINT=ROOT/'logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt'


def old_package():
    path=ARTIFACT/'before/mjlab_training/src/ct_mjlab/fudan_ppo_walk'
    spec=importlib.util.spec_from_file_location('_fudan_structure_before',path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(spec.name+'.env'),import_module(spec.name+'.contract'),import_module(spec.name+'.runner')


def backend(n=4):
    f=fixture(n)
    model=f.env.sim.mj_model
    model.joint=lambda name:model.model.joint(name.removeprefix('robot/') if isinstance(name,str) else name)
    f.robot.find_joints=lambda names, preserve_order=True:([f.robot.joint_names.index(name) for name in names],names)
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
    f.action.cfg=SimpleNamespace(disturbances=DisturbanceConfig(),task_cfg=f.cfg)
    return f


def reward_state(env,seed):
    torch.manual_seed(seed);n=env.num_envs
    model=env.env.sim.mj_model.model
    env.env.sim.data.xpos=torch.randn(n,model.nbody,3)
    env.env.sim.data.xanchor=torch.randn(n,model.njnt,3)
    env.linear=torch.randn(n,3);env.angular=torch.randn(n,3);env.gravity=torch.randn(n,3)
    env.height=torch.rand(n);env.velocity=torch.randn(n,6);env.acceleration=torch.randn(n,6)
    env.previous=torch.randn(n,6,2);env.action.torques=torch.randn(n,6);env.action.raw_action=torch.randn(n,6)
    env.commands=torch.randn(n,3)


def tensors_equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tensors_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tensors_equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);checks=[]
    def verify(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def forbidden(*args,**kw):raise AssertionError('Real physics/optimizer update forbidden')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as temporary:
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','mjlab.envs.ManagerBasedRlEnv.step','torch.optim.Adam.step'):
            stack.enter_context(patch(name,side_effect=forbidden))
        old,old_contract,old_runner=old_package()
        for version in (contract.RESET_V2_CONTRACT,contract.RESET_CONTRACT):
            b=backend();n=backend()
            torch.manual_seed(190)
            left=old.FudanEnv(4,device='cpu',physics_env=b.env,reset_contract=version)
            state_left=torch.get_rng_state()
            torch.manual_seed(190)
            right=current.FudanEnv(4,device='cpu',physics_env=n.env,reset_contract=version)
            verify(version+':constructor_rng',torch.equal(state_left,torch.get_rng_state()))
            verify(version+':limits_centres',torch.equal(left.front_limits,right.front_limits) and torch.equal(left.action.nominal_positions,right.action.nominal_positions))
            for seed in (7,39,971):
                reward_state(left,seed);reward_state(right,seed)
                ra=left.reward();rb=right.reward()
                verify(version+f':reward16_exact_{seed}',tensors_equal(ra,rb))
                torch.manual_seed(seed);oa,ca=left.observations();ha=left._history_frame.clone();rng=torch.get_rng_state()
                torch.manual_seed(seed);ob,cb=right.observations()
                verify(version+f':observations_critic_history_rng_{seed}',torch.equal(oa,ob) and torch.equal(ca,cb) and torch.equal(ha,right._history_frame) and torch.equal(rng,torch.get_rng_state()))
            # Real reset/step/command methods, with only the underlying physical backend replaced.
            left=fixture(4,reset_contract=version);right=fixture(4,reset_contract=version)
            left.__class__=old.FudanEnv
            def resample_before(self,ids):
                if len(ids):self.events.append('command_resample');self.resample_calls.append(ids.clone())
                old.FudanEnv.resample(self,ids)
            left.resample=MethodType(resample_before,left)
            torch.manual_seed(41)
            la=old_runner.initialize_rollout(left);rng=torch.get_rng_state()
            torch.manual_seed(41)
            rb=runner.initialize_rollout(right)
            verify(version+':startup_warmup_phase_rng',tensors_equal(la,rb) and torch.equal(rng,torch.get_rng_state()) and left.events==right.events)
            for tick in range(8):
                action=torch.full((4,6),float(tick)/10)
                torch.manual_seed(100+tick);la=left.step(action,auto_reset=False,resample=False);rng=torch.get_rng_state()
                torch.manual_seed(100+tick);rb=right.step(action,auto_reset=False,resample=False)
                verify(version+f':history_step_{tick}',tensors_equal(la,rb) and torch.equal(rng,torch.get_rng_state()))
            untouched=right.history[[0,2,3]].clone()
            torch.manual_seed(823);left.reset_idx(torch.tensor([1]));rng=torch.get_rng_state()
            torch.manual_seed(823);right.reset_idx(torch.tensor([1]))
            verify(version+':local_reset_isolated_rng',torch.equal(left.history,right.history) and torch.equal(right.history[[0,2,3]],untouched) and torch.equal(rng,torch.get_rng_state()))
            verify(version+':commands_sampling_identical',torch.equal(left.commands,right.commands) and torch.equal(left.command_mode,right.command_mode) and torch.equal(left.stop_command_step,right.stop_command_step))

            left.total_steps=right.total_steps=2000
            left.sums[:]=right.sums[:]=20.
            left.speed_limit=right.speed_limit=2.
            torch.manual_seed(932);left.reset_idx(torch.tensor([1]));rng=torch.get_rng_state()
            torch.manual_seed(932);right.reset_idx(torch.tensor([1]))
            verify(version+':curriculum_increment_and_rng_exact',left.speed_limit==right.speed_limit==2.1 and torch.equal(rng,torch.get_rng_state()) and torch.equal(left.commands,right.commands))

        # Current config is actually read by physical-config, reward, observation, network and PPO consumers.
        cfg=FudanWalkCfg();cfg.control.wheel_stiffness=.17;cfg.sim.dt=.004
        physical=current.make_env_cfg(task_cfg=cfg)
        verify('control_and_timestep_config_consumer',physical.actions['pd'].wheel_stiffness==.17 and physical.sim.mujoco.timestep==.004 and physical.decimation==5)
        original=current.make_env_cfg();previous=old.make_env_cfg()
        control_fields=['leg_stiffness','leg_damping','leg_position_scale','wheel_stiffness','wheel_damping','wheel_velocity_scale','max_wheel_velocity','nominal_leg_positions','action_delay_s','ground_start_hold_s']
        verify('all_physical_control_defaults_preserved',all(getattr(original.actions['pd'],key)==getattr(previous.actions['pd'],key) for key in control_fields) and original.sim.mujoco.timestep==previous.sim.mujoco.timestep and original.decimation==previous.decimation)
        f=backend();env=current.FudanEnv(4,device='cpu',physics_env=f.env);reward_state(env,39)
        base=env.reward()[1]['tracking_lin_vel'].clone();env.cfg.rewards.scales['tracking_lin_vel']=.5
        changed=env.reward()[1]['tracking_lin_vel'];verify('reward_weight_actual_consumer',torch.equal(base*.5,changed))
        env.commands[:,0]=.23;env.cfg.normalization.command[0]=3.
        verify('observation_command_scale_actual_consumer',torch.equal(env.frame()[:,6],torch.full((4,),.69)))
        isolated=FudanWalkCfg();verify('config_instances_do_not_alias',isolated.rewards.scales['tracking_lin_vel']==1. and isolated.normalization.command[0]==2.)
        train=FudanWalkCfgPPO();train.policy.init_noise_std=.55;train.algorithm.clip_param=.3;train.runner.num_steps_per_env=12
        with redirect_stdout(io.StringIO()):learner=runner.FudanRunner(env,train_cfg=train,device='cpu',log_dir=Path(temporary)/'configuration')
        verify('network_ppo_rollout_config_actual_consumers',torch.equal(learner.policy.std,torch.full((6,),.55)) and learner.alg.clip_param==.3 and learner.alg.storage.num_transitions_per_env==12)
        verify('optimizer_methods_not_replaced','step' not in learner.alg.optimizer.__dict__ and 'step' not in learner.alg.extra_optimizer.__dict__)
        learner.writer.close()

        # Same initial RNG produces exactly the same initialized network and action distribution.
        torch.manual_seed(91)
        with redirect_stdout(io.StringIO()):old_policy=old_contract.make_network()
        rng=torch.get_rng_state();torch.manual_seed(91)
        with redirect_stdout(io.StringIO()):new_policy=contract.make_network()
        verify('network_initialization_and_rng_exact',tensors_equal(old_policy.state_dict(),new_policy.state_dict()) and torch.equal(rng,torch.get_rng_state()))
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        old_policy.load_state_dict(checkpoint['model_state_dict']);new_policy.load_state_dict(checkpoint['model_state_dict'])
        torch.manual_seed(183);obs=torch.randn(32,25);history=torch.randn(32,125)
        with torch.inference_mode():
            verify('8000_actions_encoder_frozen_exact',tensors_equal(old_policy.act_inference(obs,history),new_policy.act_inference(obs,history)))
            torch.manual_seed(31);first=old_policy.act(obs,history);rng=torch.get_rng_state()
            torch.manual_seed(31);second=new_policy.act(obs,history)
            verify('action_distribution_sampling_rng_exact',torch.equal(first,second) and torch.equal(rng,torch.get_rng_state()))
        verify('frozen_model_weights_unchanged',tensors_equal(old_policy.state_dict(),checkpoint['model_state_dict']) and tensors_equal(new_policy.state_dict(),checkpoint['model_state_dict']))

        # Actual registry resolution enters the actual task and runner constructors with a nonphysical backend.
        calls=[];cfg,train=task_registry.get_cfgs('fudan_ppo_walk');fake=backend()
        with patch('mjlab.envs.ManagerBasedRlEnv',side_effect=lambda *args,**kw:(calls.append('physical_backend_replaced') or fake.env)):
            env,resolved=task_registry.make_env('fudan_ppo_walk',env_cfg=cfg,train_cfg=train,num_envs=4,device='cpu')
        verify('registry_constructs_actual_task_environment',isinstance(env,current.FudanEnv) and env.cfg is cfg and calls==['physical_backend_replaced'])
        with redirect_stdout(io.StringIO()):real_runner,_=task_registry.make_alg_runner(env,'fudan_ppo_walk',train_cfg=train,log_dir=Path(temporary)/'registry',device='cpu')
        verify('registry_constructs_actual_runner_and_ppo',isinstance(real_runner,runner.FudanRunner) and isinstance(real_runner.alg,PPO))
        real_runner.writer.close()
        # The callback preserves exact update ordering/counters with a fake optimizer, never an Adam update.
        real_runner.deadline=float('inf');fake_optimizer=SimpleNamespace(param_groups=[],step=lambda:calls.append('step'))
        real_runner.optimizer_step(fake_optimizer,'primary_updates');real_runner.optimizer_step(fake_optimizer,'encoder_updates')
        verify('actual_update_counters_after_success',real_runner.primary_updates==1 and real_runner.encoder_updates==1 and calls[-2:]==['step','step'])
        real_runner.deadline=0
        try:real_runner.optimizer_step(fake_optimizer,'primary_updates');stopped=False
        except runner.BudgetStop:stopped=True
        verify('budget_rejects_before_optimizer',stopped and real_runner.primary_updates==1 and calls[-2:]==['step','step'])

    result=dict(passed=True,count=len(checks),checks=checks,physics_steps=0,optimizer_updates=0,
        frozen_model=str(CHECKPOINT),model_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        scope='Nonphysical exact tensor/control-flow comparison and frozen CPU network forwards; not behavior acceptance')
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(passed=True,count=len(checks))))

if __name__=='__main__':main()
