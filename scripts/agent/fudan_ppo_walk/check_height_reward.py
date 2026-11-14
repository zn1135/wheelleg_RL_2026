"""Check actual reward scaling, input preservation, metadata and precheck gate without training."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ct_mjlab.envs.fudan_ppo_walk import height_reward,posture_reward
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
from ct_mjlab.envs.fudan_ppo_walk.wheel_physics import record
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner
from ct_mjlab.scripts import train
from .reset_v2_fixture import fixture


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);checks={};profile=height_reward.from_task_config()
    def check(name,value):checks[name]=bool(value);assert checks[name],name
    def forbidden(*args,**kwargs):raise AssertionError('No actual integration, policy forward or optimizer update')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as tmp:
        for name in ('mujoco.mj_step','mjlab.envs.ManagerBasedRlEnv.step','torch.nn.Linear.forward','torch.optim.Adam.step'):
            stack.enter_context(patch(name,side_effect=forbidden))
        check('default_height_scale5_sigma001',profile['scale']==5. and profile['sigma']==.01)
        check('missing_old_metadata_restores_weight1',height_reward.from_record()==height_reward.LEGACY)
        for value in ({**profile,'scale':-1.},{**profile,'sigma':0.}):
            try:height_reward.from_record(value)
            except ValueError:pass
            else:raise AssertionError('Invalid height recipe accepted')
        check('invalid_height_parameters_rejected',True)
        env=fixture(6);env.randomized=False;model=env.env.sim.mj_model.model
        posture_reward.configure(env.cfg,posture_reward.from_task_config());env.posture_reward_profile=posture_reward.from_task_config()
        env.wheel_ids=torch.tensor([model.body(n).id for n in ('lfwheel','rfwheel')])
        env.hip_ids=torch.tensor([model.joint(n).id for n in ('lf0','rf0')])
        env.front_q=torch.tensor([int(model.joint(n).qposadr[0]) for n in ('lf0','lf1','rf0','rf1')])
        env.front_limits=torch.tensor([[-100.,100.]]*4)
        env.env.sim.data.xpos=torch.zeros(6,model.nbody,3);env.env.sim.data.xanchor=torch.zeros(6,model.njnt,3)
        env.commands[:,2]=.26;env.height=env.commands[:,2]+torch.tensor([0.,.01,.05,-.075,-.115,-.155])
        height_reward.configure(env.cfg,None);before_inputs=env.observations();before_history=env._history_frame.clone()
        old_total,old=FudanEnv.reward(env)
        narrow=dict(profile,sigma=.001);height_reward.configure(env.cfg,narrow);_,scaled=FudanEnv.reward(env)
        check('weight_alone_remains_five_times',torch.allclose(scaled['base_height'],5*old['base_height'],atol=1e-9,rtol=1e-6))
        height_reward.configure(env.cfg,profile);new_total,new=FudanEnv.reward(env)
        expected=.05*torch.exp(-(env.height-env.commands[:,2]).square()/.01)
        check('actual_wider_height_reward_matches_formula',torch.allclose(new['base_height'],expected,atol=1e-8,rtol=1e-6))
        check('widening_improves_nonzero_height_errors',new['base_height'][1:].gt(scaled['base_height'][1:]).all())
        check('height_cap_really_increases_to005',torch.isclose(new['base_height'][0],torch.tensor(.05),atol=1e-8))
        check('other16_reward_terms_exactly_unchanged',len(new)==17 and all(torch.equal(v,new[k]) for k,v in old.items() if k!='base_height'))
        check('logged_total_is_sum_of_terms',torch.equal(new_total,torch.stack(list(new.values())).sum(0)))
        check('total_difference_only_from_height',torch.allclose(new_total-old_total,new['base_height']-old['base_height'],atol=1e-8))
        after_inputs=env.observations()
        check('actor_critic_and_history_unchanged',all(torch.equal(x,y) for x,y in zip(before_inputs,after_inputs)) and torch.equal(before_history,env._history_frame))
        check('low_pose_has_measurable_height_reward',.01<float(new['base_height'][4])<.02)
        check('narrow_checkpoint_keeps_original_sigma',height_reward.configure(env.cfg,narrow)==narrow and env.cfg.rewards.height_sigma==.001)
        height_reward.configure(env.cfg,profile)
        env.action.leg_load=None;env.wheel_physics_profile=record('mechanical')
        learner=object.__new__(FudanRunner);learner.env=env;learner.log_dir=Path(tmp)
        learner.policy=SimpleNamespace(state_dict=lambda:{})
        learner.alg=SimpleNamespace(optimizer=SimpleNamespace(state_dict=lambda:{}),extra_optimizer=SimpleNamespace(state_dict=lambda:{}))
        for name in ('iteration','transitions','primary_updates','encoder_updates'):setattr(learner,name,0)
        learner.ppo_config={};learner.partial_update=False;learner.initialization='metadata_check_only';learner.resume_from=None
        saved=torch.load(learner.save('metadata-only.pt'),map_location='cpu',weights_only=False)
        check('actual_checkpoint_save_records_height_recipe',saved['height_reward']==profile and saved['iterations']==0)
        proof=dict(passed=True,reset_contract=RESET_CONTRACT,kind='physical-visible-precheck',num_envs=20,physics_steps=1,
            disturbances=from_task_config().record(),command_sampling=command_config(),posture_reward=posture_reward.from_task_config())
        path=Path(tmp)/'proof.json';path.write_text(json.dumps(proof))
        argv=['--task','fudan_ppo_walk','--num-envs','20','--iterations','1','--precheck',str(path),'--log-dir',str(Path(tmp)/'never-train')]
        class GatePassed(Exception):pass
        with patch.object(train.task_registry,'make_env',side_effect=GatePassed),patch('ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.snapshot'):
            try:train.main(argv)
            except ValueError as exc:check('stale_height_precheck_rejected','height reward' in str(exc))
            else:raise AssertionError('Stale height precheck accepted')
            proof['height_reward']=narrow;path.write_text(json.dumps(proof))
            try:train.main(argv)
            except ValueError as exc:check('same_weight_narrow_precheck_rejected','height reward' in str(exc))
            else:raise AssertionError('Stale narrow precheck accepted')
            proof['height_reward']=profile;path.write_text(json.dumps(proof))
            try:train.main(argv)
            except GatePassed:check('matching_height_precheck_accepted',True)
            else:raise AssertionError('Matching height precheck did not reach environment gate')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(passed=all(checks.values()),checks=checks,height_reward=profile,physics_steps=0,policy_forwards=0,optimizer_updates=0,
        scope='Actual reward/input/save/CLI functions with synthetic state; no learned behavior validation')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
