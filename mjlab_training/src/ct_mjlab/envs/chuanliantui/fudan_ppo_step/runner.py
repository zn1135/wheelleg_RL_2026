"""新任务迁移所有网络，重建优化器；同配方恢复才使用原优化器。"""
import json
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner, sha
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import section_dict, timing_record
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.wheel_physics import record as wheel_record
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import from_record, from_task_config
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT
from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import TASK, recipe_record


def validate_source(saved, cfg):
    if (json.dumps(saved['control'], sort_keys=True) != json.dumps(section_dict(cfg.control), sort_keys=True)
            or saved['control_timing'] != timing_record(cfg)):
        raise ValueError('Initialization source physical control/timing differs')
    if saved['physical_wheel_model'] != wheel_record(cfg.sim.wheel_model):
        raise ValueError('Initialization wheel physics differs')
    if saved['reset_contract'] != RESET_CONTRACT or from_record(saved['disturbances']) != from_task_config(cfg):
        raise ValueError('Initialization source reset/history or delays differ')


def initialize_policy(policy, path, cfg):
    source_policy, saved = load_policy(path, device=next(policy.parameters()).device)
    validate_source(saved, cfg)
    policy.load_state_dict(source_policy.state_dict(), strict=True)
    assert all(torch.equal(value, source_policy.state_dict()[key]) for key, value in policy.state_dict().items())
    return dict(path=str(path), sha256=sha(path), source_iterations=saved['iterations'],
        inherited=['actor', 'encoder', 'critic', 'std'], std=policy.std.detach().cpu().tolist(),
        actor_parameter_max_difference=0., critic_reinitialized=False,
        optimizers='fresh', task_iterations=0)


class StepRunner(FudanRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.flat_teacher=None
        self.flat_preservation_last=0.

    def _load_flat_teacher(self):
        cfg=self.env.cfg.step
        if not getattr(cfg,'flat_preservation_weight',0.):return
        root=Path(__file__).resolve().parents[6]
        path=root/cfg.flat_teacher_path
        if sha(path)!=cfg.flat_teacher_sha256:
            raise ValueError('Flat preservation teacher fingerprint differs')
        devices=[torch.device(self.device).index or 0] if torch.device(self.device).type=='cuda' else []
        # 加载固定教师不消耗续训刚恢复的随机数状态。
        with torch.random.fork_rng(devices=devices):
            teacher,saved=load_policy(path,device=self.device)
        validate_source(saved,self.env.cfg)
        self.flat_teacher=teacher.eval().requires_grad_(False)

    def flat_preservation_loss(self,obs,history,flat):
        if hasattr(self.env.cfg.step,'flat_teacher_height_max'):
            # 旧教师只见过0.22～0.30m，不约束新增0.35m转向探索。
            flat=flat&(obs[:,8]<=self.env.cfg.step.flat_teacher_height_max*self.env.cfg.normalization.command[2]+1e-6)
        if self.flat_teacher is None or not flat.any():return None
        # 仅平地约束旧动作；台阶动作不受教师模仿限制。
        with torch.no_grad():
            reference,_=self.flat_teacher.act_inference(obs[flat],history[flat])
            latent=self.policy.encode(history[flat])
        current=self.policy.actor(torch.cat((obs[flat],latent),-1))
        return self.env.cfg.step.flat_preservation_weight*((current-reference)/self.flat_teacher.std).square().mean()

    def optimizer_step(self,optimizer,counter):
        if counter=='primary_updates' and self.flat_teacher is not None:
            storage=self.alg.storage
            loss=self.flat_preservation_loss(storage.observations[-1],storage.observation_history[-1],self.env.modes==0)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],self.alg.max_grad_norm)
                self.flat_preservation_last=float(loss.detach())
        return super().optimizer_step(optimizer,counter)

    def initialize_from_ppo(self, path):
        if self.iteration or self.alg.optimizer.state or self.alg.extra_optimizer.state:
            raise ValueError('Policy initialization requires a fresh runner')
        self.transfer = initialize_policy(self.policy, path, self.env.cfg)
        self._load_flat_teacher()
        self.initialization = 'all_networks_from_walk_fresh_optimizers'
        (self.log_dir/'initialization.json').write_text(json.dumps(self.transfer, indent=2)+'\n')
        return self.transfer

    def initialize_rollout(self):
        # 上台/停车是完整事件序列，不把首回合随机截短成几拍。
        return self.env.reset()

    def checkpoint_metadata(self):
        return dict(task=TASK, training_recipe=recipe_record(self.env.cfg),
            recipe_migration=getattr(self,'recipe_migration',None),
            transfer=getattr(self, 'transfer', None),
            terrain_levels=self.env.terrain.terrain_levels.cpu(),
            terrain_types=self.env.terrain.terrain_types.cpu())

    def load(self, path, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('task') != TASK or saved.get('training_recipe') != recipe_record(self.env.cfg):
            raise ValueError('Step resume requires the identical saved training recipe')
        result = super().load(path, **kwargs)
        self.transfer = saved.get('transfer')
        self.recipe_migration = saved.get('recipe_migration')
        if len(saved['terrain_levels']) != self.env.num_envs:
            raise ValueError('Step resume preserves per-environment curriculum; num_envs must match')
        self.env.terrain.terrain_levels.copy_(saved['terrain_levels'].to(self.device))
        self.env.terrain.terrain_types.copy_(saved['terrain_types'].to(self.device))
        self.env._set_origins(torch.arange(self.env.num_envs, device=self.device))
        self._load_flat_teacher()
        return result

    def log_task_metrics(self):
        if self.flat_teacher is not None:
            self.writer.add_scalar('Flat/reference_action_loss',self.flat_preservation_last,self.iteration)
        rows, self.env.episode_rows = self.env.episode_rows, []
        if hasattr(self.env, 'stop_totals'):
            for terrain, (samples, speed_sum, steady) in zip(('flat', 'crossed'), self.env.stop_totals.tolist()):
                tag = f'Stop/{terrain}'
                self.writer.add_scalar(f'{tag}_samples', samples, self.iteration)
                self.writer.add_scalar(f'{tag}_speed_m_s', speed_sum/samples if samples else float('nan'), self.iteration)
                self.writer.add_scalar(f'{tag}_steady_fraction', steady/samples if samples else float('nan'), self.iteration)
            self.env.stop_totals.zero_()
            for mode, field in ((0, 'flat_boundary_timeout'), (2, 'down_lane_failure')):
                ended = [r[field] for r in rows if r['mode'] == mode]
                self.writer.add_scalar(f'Step/{field}_rate', np.mean(ended) if ended else float('nan'), self.iteration)
            from ct_mjlab.fudan_ppo_walk.contract import MOTORS
            for name, std in zip(MOTORS, self.policy.std.detach().tolist()):
                self.writer.add_scalar(f'Policy/std_{name}', std, self.iteration)
        self.writer.add_scalar('Step/terrain_level', self.env.terrain.terrain_levels.float().mean(), self.iteration)
        self.writer.add_scalar('Step/max_terrain_level', self.env.terrain.terrain_levels.max(), self.iteration)
        # 合并已结束及在途回合，不能因失败复位把接触峰值清成零。
        metrics = self.env.contacts.metrics(slice(None))
        values = {key: value.cpu().numpy() for key, value in metrics.items()}
        if rows:
            self.writer.add_scalar('Step/ray_failure_rate',np.mean([row.get('ray_missed',False) for row in rows]),self.iteration)
            values = {key: np.concatenate((value, np.asarray([row[key] for row in rows])), axis=0)
                      for key, value in values.items()}
        for i, name in enumerate(('body', 'leg')):
            for key in ('contact_rate', 'peak_N', 'longest_s'):
                part = values[key][:, i]
                statistic = part.mean() if key == 'contact_rate' else part.max()
                self.writer.add_scalar(f'Contact/{name}/{key}', statistic, self.iteration)
            peak_index = int(values['peak_N'][:, i].argmax())
            if values['peak_N'][peak_index, i] > self.env.cfg.step.contact_threshold_N:
                point = values['peak_position_w'][peak_index, i]
                for axis, value in zip('xyz', point):
                    self.writer.add_scalar(f'Contact/{name}/peak_position_{axis}_m', value, self.iteration)
        if rows:
            with (self.log_dir/'episodes.jsonl').open('a') as stream:
                for row in rows:
                    stream.write(json.dumps(dict(iteration=self.iteration, **row))+'\n')
            step_rows = [row for row in rows if row['mode'] != 0]
            if step_rows:
                self.writer.add_scalar('Step/success_rate', np.mean([r['success'] for r in step_rows]), self.iteration)
                self.writer.add_scalar('Step/crossed_rate', np.mean([r['crossed'] for r in step_rows]), self.iteration)
                for reverse, name in ((False,'forward'),(True,'reverse')):
                    selected=[r for r in step_rows if r['reverse']==reverse]
                    if selected:
                        self.writer.add_scalar(f'Step/{name}_success_rate', np.mean([r['success'] for r in selected]), self.iteration)
                self.writer.add_scalar('Step/max_approach_speed', max(r['max_approach_speed'] for r in step_rows), self.iteration)
