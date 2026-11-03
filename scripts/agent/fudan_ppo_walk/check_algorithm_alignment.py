"""Exact PPO/storage/control comparison with optimizer commits replaced by recorders."""
import argparse,ast,hashlib,io,json
from contextlib import ExitStack,redirect_stdout
from pathlib import Path
from unittest.mock import patch
import torch
from .check_structure_alignment import old_package,backend,tensors_equal,ARTIFACT,CHECKPOINT
from ct_mjlab.envs.fudan_ppo_walk import fudan_ppo_walk as current
from ct_mjlab.fudan_ppo_walk import contract
from ct_mjlab.rsl_rl.algorithms.fudan_ppo import PPO


def definitions(path):
    return [n for n in ast.parse(path.read_text()).body if not isinstance(n,(ast.Import,ast.ImportFrom))]


class NormalizeHook(ast.NodeTransformer):
    def visit_If(self,node):
        if 'self.optimizer_step is None'==ast.unparse(node.test):return self.visit(node.body[0])
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if any(ast.unparse(t)=='self.optimizer_step' for t in node.targets):return None
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        if node.name=='__init__' and any(a.arg=='optimizer_step' for a in node.args.args):
            node.args.args.pop();node.args.defaults.pop()
        return self.generic_visit(node)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    checks=[]
    def verify(name,condition):
        assert condition,name
        checks.append(name)
    def forbidden(*args,**kw):raise AssertionError('Real integration/optimizer commit forbidden')
    with ExitStack() as stack:
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','torch.optim.Adam.step','mjlab.envs.ManagerBasedRlEnv.step'):
            stack.enter_context(patch(name,side_effect=forbidden))
        old,old_contract,old_runner=old_package()
        source=ARTIFACT/'before/mjlab_training/src/ct_mjlab/fudan_ppo_walk'
        root=Path('mjlab_training/src/ct_mjlab/rsl_rl')
        for before,after in [('network.py',root/'modules/actor_critic_sequence.py'),('storage.py',root/'storage/fudan_rollout_storage.py'),('utils.py',root/'utils.py')]:
            left=definitions(source/before);right=definitions(after)
            verify(before+':all_computations_ast_identical',ast.dump(ast.Module(body=left,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=right,type_ignores=[]),include_attributes=False))
        left=ast.Module(body=definitions(source/'ppo.py'),type_ignores=[])
        right=NormalizeHook().visit(ast.Module(body=definitions(root/'algorithms/fudan_ppo.py'),type_ignores=[]))
        verify('ppo_only_change_is_explicit_optimizer_callback',ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False))
        for seed in (91,126):
            lb,rb=backend(),backend()
            for obj in (lb,rb):
                obj.action.substep_rows=[];obj.action.record_substeps=False;obj.action._env=obj.env;obj.action.num_envs=4;obj.env.physics_dt=.002
            torch.manual_seed(seed);action=torch.randn(4,6)*150.
            old.PhysicalAction.process_actions(lb.action,action);current.PhysicalAction.process_actions(rb.action,action)
            verify(f'action_clipping_and_cache_exact_{seed}',torch.equal(lb.action._raw_actions,rb.action._raw_actions) and lb.action.substep_rows==rb.action.substep_rows)
            verify(f'sensor_pack_scale_exact_{seed}',torch.equal(old.PhysicalAction.sensor_frame(lb.action),current.PhysicalAction.sensor_frame(rb.action)))
            torch.manual_seed(seed);offset=torch.randn_like(lb.action.last_sample_q)
            lb.action.last_sample_q-=offset;rb.action.last_sample_q-=offset
            verify(f'500hz_joint_difference_exact_{seed}',torch.equal(old.sample_substep(lb.env),current.sample_substep(rb.env)) and torch.equal(lb.action.sample_velocity,rb.action.sample_velocity))
            for obj in (lb,rb):obj.env.sim.get_default_field=lambda name,m=obj.env.sim.mj_model.model:torch.as_tensor(getattr(m,name),dtype=torch.float32)
            torch.manual_seed(seed);old.randomize(lb.env,None);rng=torch.get_rng_state()
            torch.manual_seed(seed);current.randomize(rb.env,None)
            verify(f'randomization_all_arrays_and_rng_exact_{seed}',torch.equal(rng,torch.get_rng_state()) and tensors_equal(lb.env._fudan_parameters,rb.env._fudan_parameters) and all(torch.equal(getattr(lb.env.sim.model,k),getattr(rb.env.sim.model,k)) for k in ('body_mass','body_inertia','body_ipos','geom_friction','geom_solref')))

        # Execute both actual PPO loss/backward/mini-batch implementations; every optimizer commit is a no-op recorder.
        from importlib import import_module
        old_ppo=import_module('_fudan_structure_before.ppo').PPO
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        captures=[]
        for which,factory,klass in [('before',old_contract.make_network,old_ppo),('after',contract.make_network,PPO)]:
            torch.manual_seed(831)
            with redirect_stdout(io.StringIO()):policy=factory()
            policy.load_state_dict(checkpoint['model_state_dict']);alg=klass(policy,device='cpu',**contract.PPO_CONFIG);alg.init_storage(4,4,[25],[144],[125],[6])
            commits=[]
            def capture(optimizer,name):
                h=hashlib.sha256()
                for group in optimizer.param_groups:
                    for parameter in group['params']:
                        if parameter.grad is not None:h.update(parameter.grad.detach().numpy().tobytes())
                commits.append((name,h.hexdigest()))
            if which=='before':
                alg.optimizer.step=lambda:capture(alg.optimizer,'primary_updates')
                alg.extra_optimizer.step=lambda:capture(alg.extra_optimizer,'encoder_updates')
            else:alg.optimizer_step=capture
            torch.manual_seed(701)
            with torch.inference_mode():
                for t in range(4):
                    obs=torch.randn(4,25);hist=torch.randn(4,125);critic=torch.randn(4,141)
                    alg.act(obs,hist,critic)
                    alg.process_env_step(torch.randn(4),torch.zeros(4,dtype=torch.bool),{},torch.randn(4,25))
                alg.compute_returns(torch.cat((critic,policy.encode(hist)),-1))
            losses=alg.update()
            verify(which+':frozen_weights_after_mock_update',tensors_equal(policy.state_dict(),checkpoint['model_state_dict']))
            captures.append(dict(losses=losses,commits=commits,rng=torch.get_rng_state(),learning_rate=alg.learning_rate,
                primary_lr=alg.optimizer.param_groups[0]['lr'],encoder_lr=alg.extra_optimizer.param_groups[0]['lr']))
        verify('ppo_losses_gradients_update_order_rng_exact',tensors_equal(captures[0],captures[1]))
        verify('primary20_then_encoder20_recorded', [x[0] for x in captures[1]['commits']]==['primary_updates']*20+['encoder_updates']*20)
    a.output.write_text(json.dumps(dict(passed=True,count=len(checks),checks=checks,physics_steps=0,optimizer_updates=0,
        mock_commits_each_algorithm=40,scope='Actual CPU loss/backward and synthetic rollout; no Adam commits or physical training'),indent=2)+'\n')
    print(json.dumps(dict(passed=True,count=len(checks))))

if __name__=='__main__':main()
