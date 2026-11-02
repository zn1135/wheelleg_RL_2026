"""Bounded first-round runner; optimizer and physical budgets are explicit."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from .contract import CONTRACT,REFERENCE,PPO_CONFIG,make_network,RESET_CONTRACT,RESET_V2_CONTRACT,WEIGHTS
from .env import FudanEnv
from .ppo import PPO
from .disturbances import DisturbanceConfig,from_record


class BudgetStop(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize_rollout(env):
    # 对齐参考train.py的init_at_random_ep_len=True，且当前帧与历史末帧共用一次采样。
    obs,critic,history=env.reset()
    env.episode_steps.copy_(torch.randint_like(env.episode_steps,high=2000))
    env.env.episode_length_buf.copy_(env.episode_steps)
    return obs,critic,history


class FudanRunner:
    def __init__(self,env,train_cfg=None,log_dir=None,device='cuda:0'):
        self.env=env if isinstance(env,FudanEnv) else FudanEnv(physics_env=env.unwrapped,device=device)
        self.device=device;self.policy=make_network().to(device)
        self.alg=PPO(self.policy,device=device,**PPO_CONFIG)
        self.alg.init_storage(self.env.num_envs,48,[25],[144],[125],[6])
        self.log_dir=Path(log_dir);self.log_dir.mkdir(parents=True,exist_ok=True)
        self.writer=SummaryWriter(str(self.log_dir))
        self.iteration=0;self.transitions=0;self.primary_updates=0;self.encoder_updates=0
        self.initialization = 'random_from_scratch_seed_1'
        self.resume_from = None
        self.history_migration = None
        self.deadline=float('inf');self.partial_update=False
        for optimizer,counter in ((self.alg.optimizer,'primary_updates'),(self.alg.extra_optimizer,'encoder_updates')):
            original=optimizer.step
            def step(*args,_original=original,_counter=counter,_optimizer=optimizer,**kwargs):
                if time.time()>=self.deadline:raise BudgetStop()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in _optimizer.param_groups for p in g['params']):
                    raise FloatingPointError('Non-finite optimizer gradient')
                result=_original(*args,**kwargs);setattr(self,_counter,getattr(self,_counter)+1);return result
            optimizer.step=step

    def load(self, path, *, migrate_history_v2=False,allow_disturbance_change=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        source_disturbances=from_record(checkpoint.get('disturbances'))
        target_disturbances=getattr(self.env,'disturbance_config',DisturbanceConfig())
        if source_disturbances!=target_disturbances and not allow_disturbance_change:
            raise ValueError('Disturbance settings differ; use config.py with a matching precheck through the training entry')
        target_contract=getattr(self.env,'reset_contract',RESET_CONTRACT)
        source_contract=checkpoint.get('reset_contract')
        migration=(migrate_history_v2 and source_contract==RESET_V2_CONTRACT and target_contract==RESET_CONTRACT)
        if migrate_history_v2 and not migration:
            raise ValueError('Explicit history migration only supports PD-centred v2 to noisy5/clipped v3')
        if (checkpoint['contract'] != CONTRACT or (source_contract != target_contract and not migration)
                or checkpoint['reference'] != REFERENCE or checkpoint['config'] != PPO_CONFIG):
            raise ValueError('Checkpoint recipe or reset contract differs from this training task')
        if checkpoint['partial_update']:
            raise ValueError('Cannot resume a partial rollout/update checkpoint')
        iterations = checkpoint['iterations']
        if (iterations <= 0 or checkpoint['primary_updates'] != iterations * 20
                or checkpoint['encoder_updates'] != iterations * 20):
            raise ValueError('Checkpoint update counters differ from this training task')
        source_num_envs = checkpoint.get('num_envs')
        if source_num_envs is None:
            if checkpoint['initialization'] != 'random_from_scratch_seed_1':
                raise ValueError('Legacy resumed checkpoint lacks environment/clock metadata')
            source_num_envs = checkpoint['transitions'] // (iterations * 48)
            if source_num_envs <= 0 or checkpoint['transitions'] != iterations * source_num_envs * 48:
                raise ValueError('Legacy checkpoint rollout counters are inconsistent')
        self.policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.alg.extra_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        # 自适应学习率还保存在PPO对象中，不能只恢复Adam的参数组。
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
        self.iteration = iterations
        self.transitions = checkpoint['transitions']
        self.primary_updates = checkpoint['primary_updates']
        self.encoder_updates = checkpoint['encoder_updates']
        self.env.speed_limit = checkpoint['speed_limit']
        # 旧v2存档未保存全局拍数；其冷启恰有一次热步，可由转移数还原。
        self.env.total_steps = checkpoint.get('policy_steps', self.transitions // source_num_envs + 1)
        self.env.env.common_step_counter = self.env.total_steps
        self.initialization = 'resumed_learning_state_fresh_environment'
        self.resume_from = dict(path=str(Path(path).resolve()), sha256=sha(path), iterations=iterations)
        self.history_migration=(dict(source_reset_contract=source_contract,target_reset_contract=target_contract,
            source=self.resume_from,changed=['initial_history_fill','history_clip_order'],
            learning_state_reset=False,physical_reset_changed=False) if migration else checkpoint.get('history_migration'))
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
        restored = dict(source=self.resume_from, iterations=self.iteration, transitions=self.transitions,
            primary_updates=self.primary_updates, encoder_updates=self.encoder_updates,
            learning_rate=self.alg.learning_rate, speed_limit=self.env.speed_limit,
            source_num_envs=source_num_envs, num_envs=self.env.num_envs,
            policy_steps=self.env.total_steps, environment_reinitialized=True,
            source_reset_contract=source_contract,target_reset_contract=target_contract,
            history_migration=self.history_migration,
            source_disturbances=source_disturbances.record(),disturbances=target_disturbances.record(),
            restoration='Learning state and RNG restored; physics/history/episode state restarts')
        (self.log_dir / 'restore_report.json').write_text(json.dumps(restored, indent=2) + '\n')
        return restored

    def save(self,name):
        path=self.log_dir/name
        torch.save(dict(contract=CONTRACT,reset_contract=getattr(self.env,'reset_contract',RESET_CONTRACT),reference=REFERENCE,model_state_dict=self.policy.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(),encoder_optimizer_state_dict=self.alg.extra_optimizer.state_dict(),
            iterations=self.iteration,transitions=self.transitions,primary_updates=self.primary_updates,
            encoder_updates=self.encoder_updates,partial_update=self.partial_update,speed_limit=self.env.speed_limit,
            config=PPO_CONFIG,initialization=self.initialization,resume_from=self.resume_from,
            history_migration=getattr(self,'history_migration',None),
            disturbances=getattr(self.env,'disturbance_config',DisturbanceConfig()).record(),
            num_envs=self.env.num_envs,policy_steps=self.env.total_steps,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)
        return path

    def learn_until(self,deadline,max_iterations=None):
        initial_iteration = self.iteration
        initial_transitions = self.transitions
        initial_primary_updates = self.primary_updates
        initial_encoder_updates = self.encoder_updates
        self.deadline=deadline;start=time.time();last_iteration_s=5.;self.save('model_initial.pt')
        if time.time()>=deadline:raise BudgetStop()
        obs,critic,history=initialize_rollout(self.env)
        stage_next=start+600.
        reward_names = tuple(WEIGHTS)
        try:
            while ((max_iterations is None or self.iteration<max_iterations)
                    and time.time()+max(5.,last_iteration_s*1.3)<deadline):
                tick=time.time()
                reward_sum = obs.new_zeros(())
                term_sums = obs.new_zeros(len(reward_names))
                with torch.inference_mode():
                    for _ in range(48):
                        if time.time()>=deadline:raise BudgetStop()
                        actions=self.alg.act(obs,history,critic)
                        obs,critic,reward,done,info,history=self.env.step(actions)
                        self.alg.process_env_step(reward,done,info,obs)
                        # 累计实际加权、裁剪后的贡献，覆盖本轮48步及全部环境。
                        reward_sum += reward.mean()
                        term_sums += torch.stack([info['terms'][name] for name in reward_names]).mean(dim=1)
                        self.transitions+=self.env.num_envs
                    self.alg.compute_returns(torch.cat((critic,self.policy.encode(history)),-1))
                losses=self.alg.update();self.iteration+=1;last_iteration_s=time.time()-tick
                self.writer.add_scalar('Loss/value',losses[0],self.iteration)
                self.writer.add_scalar('Loss/policy',losses[1],self.iteration)
                self.writer.add_scalar('Loss/velocity_estimator',losses[3],self.iteration)
                mean_reward = (reward_sum / 48).item()
                term_means = (term_sums / 48).tolist()
                self.writer.add_scalar('Train/reward',mean_reward,self.iteration)
                for name, value in zip(reward_names, term_means):
                    self.writer.add_scalar(f'Rewards/{name}', value, self.iteration)
                self.writer.add_scalar('Train/speed_limit',self.env.speed_limit,self.iteration)
                self.writer.add_scalar('Train/learning_rate',self.alg.learning_rate,self.iteration)
                if self.iteration == initial_iteration + 1:
                    self.writer.flush()
                if self.iteration == initial_iteration + 1 or self.iteration%10==0:
                    print(json.dumps(dict(iteration=self.iteration,transitions=self.transitions,
                        primary_updates=self.primary_updates,encoder_updates=self.encoder_updates,
                        elapsed_s=time.time()-start,iteration_s=last_iteration_s,reward=mean_reward)),flush=True)
                if self.iteration%100==0:self.save(f'model_{self.iteration}.pt')
                if time.time()>=stage_next:
                    self.save(f'stage_{self.iteration}.pt');stage_next+=600.
            self.partial_update=False
        except BudgetStop:
            self.partial_update=True
        finally:
            path=self.save('model_final.pt');self.writer.flush();self.writer.close()
        report=dict(training_wall_s=time.time()-start,iterations=self.iteration,transitions=self.transitions,
            primary_updates=self.primary_updates,encoder_updates=self.encoder_updates,partial_update=self.partial_update,
            checkpoint=str(path),checkpoint_sha256=sha(path),num_envs=self.env.num_envs,
            transitions_per_iteration=self.env.num_envs*48,minibatch=self.env.num_envs*12,
            stopped_by=('iteration_budget' if max_iterations is not None and self.iteration>=max_iterations
                else 'fixed_wall_clock_budget'),
            iteration_budget=None if max_iterations is None else max_iterations-initial_iteration,
            target_iteration=max_iterations,initial_iterations=initial_iteration,iterations_added=self.iteration-initial_iteration,
            transitions_added=self.transitions-initial_transitions,
            primary_updates_added=self.primary_updates-initial_primary_updates,
            encoder_updates_added=self.encoder_updates-initial_encoder_updates,
            reset_contract=getattr(self.env,'reset_contract',RESET_CONTRACT),history_migration=getattr(self,'history_migration',None),
            disturbances=getattr(self.env,'disturbance_config',DisturbanceConfig()).record(),
            resume_from=self.resume_from,behavior_accepted=False,
            reward_logging='weighted_clipped_mean_over_48_steps_and_all_environments')
        (self.log_dir/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report


def snapshot(output):
    output=Path(output);source=Path(__file__).resolve().parents[1]
    shutil.copytree(source,output/'source/ct_mjlab',ignore=shutil.ignore_patterns('__pycache__'))
    root=source.parents[2]
    paths=[*source.rglob('*.py'),root/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml',
        root/'docs/mjlab-fudan-ppo-walk.md']
    hashes={str(p.relative_to(root)):sha(p) for p in paths}
    (output/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
