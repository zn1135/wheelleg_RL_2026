"""Check real learner-state restoration and additional-iteration limits without physics."""
import argparse
from contextlib import ExitStack, redirect_stdout
import gc
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import torch

from ct_mjlab.scripts import train as cli
from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as module
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    restored_devices = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Physical and optimizer steps are forbidden')

    with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
        for name in ('mj_step', 'mj_step1', 'mj_step2'):
            stack.enter_context(patch.object(mujoco, name, forbidden))
        stack.enter_context(patch.object(torch.optim.Adam, 'step', forbidden))
        for device in ('cpu', 'cuda:0'):
            env = object.__new__(FudanEnv)
            env.reset_contract = original['reset_contract']
            env.num_envs = 4096
            env.speed_limit = 2.
            env.total_steps = 0
            env.env = SimpleNamespace(common_step_counter=0)
            with redirect_stdout(io.StringIO()):
                learner = module.FudanRunner(env, log_dir=Path(temporary)/device.replace(':', '_'), device=device)
                if not hasattr(learner, 'train_cfg'):
                    from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                    runner_configuration(learner)
                restored = learner.load(args.checkpoint)
            equal(learner.policy.state_dict(), original['model_state_dict'])
            equal(learner.alg.optimizer.state_dict(), original['optimizer_state_dict'])
            equal(learner.alg.extra_optimizer.state_dict(), original['encoder_optimizer_state_dict'])
            equal(torch.get_rng_state(), original['torch_rng'])
            equal(torch.cuda.get_rng_state_all(), original['cuda_rng'])
            assert learner.alg.learning_rate == original['optimizer_state_dict']['param_groups'][0]['lr']
            assert restored['source_num_envs'] == 512 and restored['num_envs'] == 4096
            assert learner.iteration == 3000 and learner.primary_updates == learner.encoder_updates == 60000
            assert env.speed_limit == 2.5 and env.total_steps == env.env.common_step_counter == 144001
            assert learner.alg.storage.step == 0 and learner.alg.storage.num_envs == 4096
            if not hasattr(learner, 'train_cfg'):
                from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                runner_configuration(learner)
            saved = torch.load(learner.save('loaded.pt'), map_location='cpu', weights_only=False)
            for key in ('model_state_dict', 'optimizer_state_dict', 'encoder_optimizer_state_dict', 'torch_rng', 'cuda_rng'):
                equal(saved[key], original[key])
            assert saved['num_envs'] == 4096 and saved['policy_steps'] == 144001
            assert saved['initialization'] == 'resumed_learning_state_fresh_environment'
            for name, changes in [('partial', {'partial_update': True}), ('reset', {'reset_contract': 'wrong'}),
                    ('counter', {'primary_updates': 59999})]:
                path = Path(temporary)/(name+'.pt')
                torch.save(dict(original, **changes), path)
                try:
                    if not hasattr(learner, 'train_cfg'):
                        from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                        runner_configuration(learner)
                    learner.load(path)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid checkpoint accepted: '+name)
            learner.writer.close()
            restored_devices.append(device)
            del learner, env, saved
            gc.collect()
            torch.cuda.empty_cache()

        # 真实训练循环从3000走到6000，采样/更新用替代实现，不执行真实Adam。
        r = object.__new__(module.FudanRunner)
        r.log_dir = Path(temporary)/'loop'
        r.log_dir.mkdir()
        r.iteration = 3000
        r.transitions = 73728000
        r.primary_updates = r.encoder_updates = 60000
        r.resume_from = {'iterations': 3000}
        r.partial_update = False
        obs, critic, history = torch.zeros(4,25), torch.zeros(4,141), torch.zeros(4,125)
        reward, done, actions = torch.zeros(4), torch.zeros(4,dtype=torch.bool), torch.zeros(4,6)
        info = {'terms': {name: torch.zeros(4) for name in module.WEIGHTS}}
        steps = 0
        logged_steps = []
        def step(action):
            nonlocal steps
            steps += 1
            return obs, critic, reward, done, info, history
        def update():
            r.primary_updates += 20
            r.encoder_updates += 20
            return [0.,0.,0.,0.]
        def save(name):
            path = r.log_dir/name
            path.write_text(str(r.iteration))
            return path
        def scalar(tag, value, iteration):
            if tag == 'Rewards/collision':
                logged_steps.append(iteration)
        r.save = save
        r.env = SimpleNamespace(num_envs=4096, speed_limit=2.5, step=step)
        r.alg = SimpleNamespace(act=lambda *args:actions,process_env_step=lambda *args:None,
            compute_returns=lambda *args:None,update=update,learning_rate=.0000759375)
        r.policy = SimpleNamespace(encode=lambda h:torch.zeros(4,3))
        r.writer = SimpleNamespace(add_scalar=scalar,flush=lambda:None,close=lambda:None)
        with patch.object(module,'initialize_rollout',return_value=(obs,critic,history)),redirect_stdout(io.StringIO()):
            if not hasattr(r, 'train_cfg'):
                from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                runner_configuration(r)
            report = r.learn_until(float('inf'), max_iterations=6000)
        assert report['iterations'] == 6000 and report['iterations_added'] == report['iteration_budget'] == 3000
        assert report['transitions_added'] == 589824000 and report['transitions'] == 663552000
        assert report['primary_updates_added'] == report['encoder_updates_added'] == 60000
        assert report['primary_updates'] == report['encoder_updates'] == 120000 and steps == 144000
        assert logged_steps == list(range(3001,6001)) and not report['partial_update']
        entered = {}
        class EntryRunner:
            def __init__(self,*args,**kwargs):self.iteration=0
            def load(self,path):
                self.iteration=3000
                entered['checkpoint']=str(path)
                return {}
            def learn_until(self,deadline,max_iterations=None):
                entered.update(deadline=deadline,target=max_iterations)
                return {}
        with patch('ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk.FudanEnv',return_value=SimpleNamespace(close=lambda:None)),patch('ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.FudanRunner',EntryRunner),patch('ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.snapshot'),redirect_stdout(io.StringIO()):
            proof=Path(temporary)/'fixture-proof.json'
            proof.write_text(json.dumps(dict(passed=True,reset_contract=module.RESET_CONTRACT,kind='physical-visible-precheck',num_envs=20,physics_steps=1000,command_sampling=__import__('ct_mjlab.envs.fudan_ppo_walk.commands',fromlist=['command_config']).command_config())))
            cli.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations','3000','--resume',str(args.checkpoint),
                '--precheck',str(proof),
                '--log-dir',str(Path(temporary)/'entry')])
        assert entered['target']==6000 and entered['deadline']==float('inf')
    result = dict(passed=True,physics_steps=0,optimizer_updates=0,model_forwards=0,
        restored_devices=restored_devices,model_and_both_adam_states_identical=True,
        adaptive_learning_rate_restored=True,cpu_and_cuda_rng_identical=True,
        source_num_envs=512,target_num_envs=4096,speed_limit=2.5,policy_clock_restored=144001,
        additional_iterations=3000,target_iteration=6000,additional_transitions=589824000,
        final_transitions=663552000,additional_updates_each=60000,final_updates_each=120000,
        scalar_steps=[3001,6000],legacy_or_partial_or_bad_counter_rejected=True,
        full_physics_state_restored=False,environment_restart_explicit=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
