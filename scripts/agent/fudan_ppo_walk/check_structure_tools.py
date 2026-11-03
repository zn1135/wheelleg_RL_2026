"""Entry, compatibility, source provenance and tool reuse checks; no physics/inference."""
import argparse,ast,hashlib,io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack,redirect_stdout
from unittest.mock import patch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, FudanWalkCfgPPO, section_dict
from ct_mjlab.fudan_ppo_walk import __main__ as legacy,contract,export,policy
from ct_mjlab.scripts import train
from ct_mjlab.fudan_ppo_walk.source_migration import verify_migration,historical_path_matches
from .check_structure_alignment import ARTIFACT


def code(path):
    return ast.parse(path.read_text())


def function(tree,name):
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
    def verify(name,condition):
        assert condition,name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('No physical/model/update operation allowed')
    before=ARTIFACT/'before';root=Path('mjlab_training/src/ct_mjlab')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as temporary:
        for name in ('mujoco.mj_step','mjlab.envs.ManagerBasedRlEnv.step','torch.nn.Linear.forward','torch.optim.Adam.step'):
            stack.enter_context(patch(name,side_effect=forbidden))
        # Prove actual unified entry routes through both registry constructors and the requested task configs.
        cfg,training=task_registry.get_cfgs('fudan_ppo_walk');verify('registered_task_config_types',isinstance(cfg,FudanWalkCfg) and isinstance(training,FudanWalkCfgPPO))
        evidence=[]
        class Runner:
            iteration=8000
            def learn_until(self,deadline,max_iterations=None):evidence.append(('learn',deadline,max_iterations));return {}
        def make_env(name,**kw):
            evidence.append(('make_env',name,kw['env_cfg'].reset_contract,kw['train_cfg'].runner.num_steps_per_env));return SimpleNamespace(close=lambda:evidence.append(('close',))),kw['env_cfg']
        def make_runner(env,name,**kw):evidence.append(('make_runner',name,section_dict(kw['train_cfg'].algorithm)));return Runner(),kw['train_cfg']
        directory=Path(temporary);proof=directory/'proof.json'
        from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
        proof.write_text(json.dumps(dict(passed=True,reset_contract=contract.RESET_CONTRACT,kind='physical-visible-precheck',num_envs=20,physics_steps=1000,command_sampling=command_config())))
        with patch.object(task_registry,'make_env',side_effect=make_env),patch.object(task_registry,'make_alg_runner',side_effect=make_runner),patch('ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.snapshot'),redirect_stdout(io.StringIO()):
            train.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations','1','--log-dir',str(directory/'run'),'--precheck',str(proof)])
        verify('actual_train_entry_registry_runner_learn_chain',[r[0] for r in evidence]==['make_env','make_runner','learn','close'] and evidence[2][2]==8001 and evidence[0][3]==48)
        with patch.object(train,'main',return_value='same_chain') as entry:
            result=legacy.main(['train','--iterations','1'])
        verify('legacy_train_routes_to_same_registered_entry',result=='same_chain' and entry.call_args.args[0]==['--task','fudan_ppo_walk','--iterations','1'])
        for name in ('env','runner','ppo','network','storage','commands','disturbances','parking_integral','reset_geometry'):
            tree=code(root/'fudan_ppo_walk'/(name+'.py'))
            verify('legacy_import_only:'+name,not any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in tree.body))
        # Other task implementations and the original SMP train body are preserved.
        old=function(code(before/root/'scripts/train.py'),'train');new=function(code(root/'scripts/train.py'),'train')
        new.body=new.body[1:]
        verify('smp_train_body_ast_identical',ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False))
        for file in ['fudan_ppo_walk/parking_integral.py']:
            old_tree=code(before/root/file);new_tree=code(root/'envs'/file)
            left=[n for n in old_tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
            right=[n for n in new_tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
            verify('integral_component_computations_ast_identical',ast.dump(ast.Module(body=left,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=right,type_ignores=[]),include_attributes=False))
        script=Path('scripts/agent/fudan_ppo_walk/diagnose_parking_integral.py')
        for name in ('restore_initial','policy_tick'):
            verify('integral_r2_'+name+'_ast_identical',ast.dump(function(code(before/script),name),include_attributes=False)==ast.dump(function(code(script),name),include_attributes=False))
        import scripts.agent.fudan_ppo_walk.precheck as precheck
        import scripts.agent.fudan_ppo_walk.evaluate as evaluate
        from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
        verify('precheck_and_evaluation_use_production_environment',precheck.FudanEnv is FudanEnv and any(isinstance(n,ast.ImportFrom) and n.module=='ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk' for n in ast.walk(code(Path(evaluate.__file__)))))
        verify('export_uses_same_policy_and_frozen_interface',export.load_policy is policy.load_policy and export.history_metadata(contract.RESET_V2_CONTRACT)['history_before_clip'])
        # Activation drift is rejected rather than silently changing inference for the v1 model.
        with patch.object(FudanWalkCfgPPO.policy,'activation','relu'):
            try:contract.validate_network_contract();rejected=False
            except ValueError:rejected=True
        verify('historical_network_contract_not_dynamic',rejected)
        record=verify_migration()
        verify('all_enumerated_current_and_old_hashes_verified',bool(record['files']))
        item=record['files'][0]
        verify('unknown_source_hash_rejected',not historical_path_matches(item['old_path'],'0'*64))
    a.output.write_text(json.dumps(dict(passed=True,count=len(checks),checks=checks,physics_steps=0,model_forwards=0,optimizer_updates=0),indent=2)+'\n')
    print(json.dumps(dict(passed=True,count=len(checks))))

if __name__=='__main__':main()
