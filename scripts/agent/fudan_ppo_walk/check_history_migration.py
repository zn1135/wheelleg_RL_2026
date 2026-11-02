"""Verify explicit v2->v3 learner-state migration without physics, forwards or updates."""
import argparse
import ast
from contextlib import ExitStack,redirect_stdout
import gc
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.fudan_ppo_walk import runner as module
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT,RESET_V2_CONTRACT,LEGACY_RESET_CONTRACT
from ct_mjlab.fudan_ppo_walk.env import FudanEnv
from .check_resume import equal


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--baseline-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert original['reset_contract']==RESET_V2_CONTRACT and original['iterations']==7000 and original['num_envs']==4096
    source_hash=module.sha(args.checkpoint)
    assert source_hash=='ea5dac38af946359321e415aa661bc78cc929ddf8f29d46fec6f0116c610c9d2'
    devices=[];restorations=[];rejections=[]
    def forbidden(*a,**kw):raise AssertionError('Physical/policy/optimizer operation forbidden')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2',
                'mjlab.envs.ManagerBasedRlEnv.step','torch.optim.Adam.step','torch.nn.Linear.forward'):
            stack.enter_context(patch(name,side_effect=forbidden))
        for device in ('cpu','cuda:0'):
            env=object.__new__(FudanEnv);env.reset_contract=RESET_CONTRACT
            env.num_envs=4096;env.speed_limit=2.;env.total_steps=0;env.env=SimpleNamespace(common_step_counter=0)
            with redirect_stdout(io.StringIO()):learner=module.FudanRunner(env,log_dir=directory/device.replace(':','_'),device=device)
            try:learner.load(args.checkpoint)
            except ValueError:pass
            else:raise AssertionError('Old history silently migrated without explicit flag')
            restored=learner.load(args.checkpoint,migrate_history_v2=True)
            equal(learner.policy.state_dict(),original['model_state_dict'])
            equal(learner.alg.optimizer.state_dict(),original['optimizer_state_dict'])
            equal(learner.alg.extra_optimizer.state_dict(),original['encoder_optimizer_state_dict'])
            equal(torch.get_rng_state(),original['torch_rng']);equal(torch.cuda.get_rng_state_all(),original['cuda_rng'])
            assert learner.alg.learning_rate==original['optimizer_state_dict']['param_groups'][0]['lr']
            assert (learner.iteration,learner.transitions,learner.primary_updates,learner.encoder_updates)==(7000,860160000,140000,140000)
            assert env.total_steps==env.env.common_step_counter==original['policy_steps'] and env.speed_limit==original['speed_limit']
            assert learner.alg.storage.step==0 and learner.alg.storage.num_envs==4096
            assert restored['source_reset_contract']==RESET_V2_CONTRACT and restored['target_reset_contract']==RESET_CONTRACT
            assert restored['history_migration']['changed']==['initial_history_fill','history_clip_order']
            restorations.append(dict(device=device,**restored))
            temporary_model=learner.save('migration-fixture-not-trained.pt')
            saved=torch.load(temporary_model,map_location='cpu',weights_only=False)
            assert saved['reset_contract']==RESET_CONTRACT and saved['resume_from']['sha256']==source_hash
            for key in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng'):
                equal(saved[key],original[key])
            assert saved['policy_steps']==original['policy_steps'] and saved['num_envs']==4096
            # 新版本同合同再次恢复仍严格保持学习状态，不需要重复迁移旗标。
            again=learner.load(temporary_model)
            assert again['history_migration']==restored['history_migration']
            try:learner.load(temporary_model,migrate_history_v2=True)
            except ValueError:pass
            else:raise AssertionError('Migration flag accepted on already-v3 checkpoint')
            cases=[('legacy',dict(original,reset_contract=LEGACY_RESET_CONTRACT)),
                ('missing_contract',{k:v for k,v in original.items() if k!='reset_contract'}),
                ('unknown',dict(original,reset_contract='unknown')),
                ('foreign_interface',dict(original,contract='wrong')),
                ('reference',dict(original,reference='wrong')),
                ('recipe',dict(original,config=dict(original['config'],learning_rate=.9))),
                ('partial',dict(original,partial_update=True)),
                ('counter',dict(original,primary_updates=139999))]
            for name,value in cases:
                path=directory/(device.replace(':','_')+'_'+name+'.pt');torch.save(value,path)
                try:learner.load(path,migrate_history_v2=True)
                except ValueError:pass
                else:raise AssertionError('Unsupported migration accepted: '+name)
                rejections.append(device+':'+name)
            equal(learner.policy.state_dict(),original['model_state_dict'])
            equal(learner.alg.optimizer.state_dict(),original['optimizer_state_dict'])
            equal(learner.alg.extra_optimizer.state_dict(),original['encoder_optimizer_state_dict'])
            equal(torch.get_rng_state(),original['torch_rng']);equal(torch.cuda.get_rng_state_all(),original['cuda_rng'])
            learner.writer.close();devices.append(device)
            del learner,env,saved
            gc.collect();torch.cuda.empty_cache()
    current=Path(module.__file__).parent
    comparisons={}
    for name in ('reset_geometry.py','ppo.py','protocol.py'):
        assert (current/name).read_bytes()==(args.baseline_source/name).read_bytes(),name
        comparisons[name]='byte_identical'
    def node(path,name):
        return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    for name in ('PhysicalAction','reset_pose','randomize','make_env_cfg','frame','_update_state','reward','resample','reset'):
        # reset同名时取首个PhysicalAction.reset；FudanEnv.reset另行按类选择。
        assert ast.dump(node(current/'env.py',name),include_attributes=False)==ast.dump(node(args.baseline_source/'env.py',name),include_attributes=False),name
        comparisons['env.py:'+name]='ast_identical'
    for name in ('reset','_update_state','frame','reward','resample'):
        left=node(current/'env.py','FudanEnv');right=node(args.baseline_source/'env.py','FudanEnv')
        fn=lambda cls:next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(fn(left),include_attributes=False)==ast.dump(fn(right),include_attributes=False),name
    for name in ('controls','row','step'):
        assert ast.dump(node(current/'native.py',name),include_attributes=False)==ast.dump(node(args.baseline_source/'native.py',name),include_attributes=False),name
        comparisons['native.py:'+name]='ast_identical'
    for name in ('WEIGHTS','PPO_CONFIG','make_network'):
        def definition(path):
            return next(n for n in ast.parse(path.read_text()).body if
                isinstance(n,ast.FunctionDef) and n.name==name or
                isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert ast.dump(definition(current/'contract.py'),include_attributes=False)==ast.dump(definition(args.baseline_source/'contract.py'),include_attributes=False),name
        comparisons['contract.py:'+name]='ast_identical'
    assert module.sha(args.checkpoint)==source_hash
    result=dict(passed=True,source_checkpoint=str(args.checkpoint),source_checkpoint_sha256=source_hash,
        source_reset_contract=RESET_V2_CONTRACT,target_reset_contract=RESET_CONTRACT,devices=devices,
        model_both_adam_learning_rate_rng_counters_identical=True,
        implicit_migration_rejected=True,already_v3_migration_flag_rejected=True,negative_cases=rejections,
        source_invariants=comparisons,restorations=restorations,physics_steps=0,model_forwards=0,optimizer_updates=0,
        temporary_fixture_models_deleted=True,new_trained_model_created=False,physical_precheck='not_run',training='not_started')
    with args.output.open('x') as output:output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_invariants','restorations','negative_cases')},indent=2))


if __name__=='__main__':main()
