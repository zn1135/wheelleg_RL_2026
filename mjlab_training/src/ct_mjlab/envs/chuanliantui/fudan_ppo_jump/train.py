"""独立模型、预算和日志；使用共用的复旦 PPO 采样和优化器。"""
from datetime import datetime
import json
import time
import torch
from ct_mjlab.fudan_ppo_walk.contract import make_network
from ct_mjlab.envs import task_registry
from ct_mjlab.utils.task_registry import LOG_ROOT
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner, snapshot
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import (TASK, LANDING_RECIPE, STOP_SINK_RECIPE, PRELOAD_RECIPE,
                     PEAK_RECIPE, ADDITIVE_PEAK_RECIPE, jump_config, recipe_record)


def load_jump_policy(checkpoint, device="cpu"):
    jump_saved = torch.load(checkpoint,map_location=device,weights_only=False)
    saved_recipe = jump_saved.get('jump_recipe')
    recipe = recipe_record(jump_config(landing=saved_recipe==LANDING_RECIPE,
                                      stop_sink=saved_recipe==STOP_SINK_RECIPE,
                                      preload=saved_recipe==PRELOAD_RECIPE, peak=saved_recipe==PEAK_RECIPE,
                                      peak_additive=saved_recipe==ADDITIVE_PEAK_RECIPE))
    if (jump_saved.get('contract') != JumpRunner.contract or jump_saved.get('task') != TASK or
            jump_saved.get('jump_recipe') != recipe['version'] or jump_saved.get('partial_update') or
            jump_saved.get('jump_config') != recipe):
        raise ValueError('Expected a complete matching Jump checkpoint')
    jump = make_network().to(device).eval()
    jump.load_state_dict(jump_saved['model_state_dict'])
    return jump, jump_saved


class JumpRunner(FudanRunner):
    contract = 'fudan-jump-physical25-history125-v1'

    def load(self, path, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        target = recipe_record(self.env.cfg)
        source = saved.get('jump_config')
        landing_migration = target['version'] == LANDING_RECIPE and source == recipe_record(jump_config())
        stop_sink_migration = target['version'] == STOP_SINK_RECIPE and source == recipe_record(jump_config(landing=True))
        preload_migration = target['version'] == PRELOAD_RECIPE and source == recipe_record(jump_config(stop_sink=True))
        peak_migration = target['version'] in (PEAK_RECIPE, ADDITIVE_PEAK_RECIPE) and source == recipe_record(jump_config(preload=True))
        migration = landing_migration or stop_sink_migration or preload_migration or peak_migration
        if source != target and not migration:
            raise ValueError('Jump checkpoint configuration differs')
        result = super().load(path, **kwargs)
        if migration:
            result['jump_recipe_migration'] = dict(source=source['version'], target=target['version'],
                optimizer_states_restored=True)
            (self.log_dir/'restore_report.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def height_profile(self):
        return None  # 此任务只有腾空高度奖励。

    def checkpoint_metadata(self):
        recipe = recipe_record(self.env.cfg)
        return dict(**super().checkpoint_metadata(), task=TASK, jump_recipe=recipe['version'],
                    command_scale=self.env.cfg.normalization.command, jump_config=recipe)

    def log_task_metrics(self):
        takeoffs, landings, duration, peak, stable = self.env.totals.tolist()
        self.writer.add_scalar('Jump/takeoffs_total', takeoffs, self.iteration)
        self.writer.add_scalar('Jump/landings_total', landings, self.iteration)
        if self.env.train_peak:
            self.writer.add_scalar('Jump/highest_landed_peak_m', self.env.highest_landed_peak.item(), self.iteration)
        if landings:
            for name, value in [('airtime_s', duration), ('peak_wheel_clearance_m', peak),
                                ('stable_landing_rate', stable)]:
                self.writer.add_scalar('Jump/'+name, value/landings, self.iteration)


def train(args):
    if any(getattr(args, name, None) is not None for name in
           ('ppo_init', 'bc_init', 'smp_prior', 'smp_weight', 'wheel_std', 'viewer')) or any(
           (args.phase_mix, args.bc_finetune, args.preserve_history_v2, args.migrate_history_v2)):
        raise ValueError('Jump starts from scratch or resumes a Jump checkpoint')
    if (args.iterations is None) == (args.deadline is None):
        raise ValueError('Supply one positive --iterations or future --deadline')
    if args.iterations is not None and args.iterations <= 0 or args.deadline is not None and not args.deadline > time.time():
        raise ValueError('Invalid Jump budget')
    if not args.device.startswith('cuda') or not args.num_envs or args.num_envs < 4 or args.num_envs % 4:
        raise ValueError('Use CUDA and a positive --num-envs divisible by 4')
    if args.precheck is None:
        raise ValueError('Supply the Jump GPU --precheck report')
    cfg, ppo = task_registry.get_cfgs(TASK)
    if getattr(args, 'jump_peak_additive', False):
        cfg = jump_config(peak_additive=True)
    elif getattr(args, 'jump_peak', False):
        cfg = jump_config(peak=True)
    elif getattr(args,'jump_preload',False):
        cfg = jump_config(preload=True)
    elif getattr(args,'jump_stop_sink',False):
        cfg = jump_config(stop_sink=True)
    elif getattr(args,'jump_landing',False):
        cfg = jump_config(landing=True)
    recipe = recipe_record(cfg)
    proof = json.loads(args.precheck.read_text())
    if not (proof.get('passed') and proof.get('task') == TASK and proof.get('recipe') == recipe['version']
            and proof.get('config') == recipe and proof.get('worlds') == 20 and proof.get('optimizer_updates') == 0):
        raise ValueError('Expected the matching 20-world Jump precheck')
    log = args.log_dir or LOG_ROOT / TASK / datetime.now().strftime('%Y%m%d_%H%M%S')
    log.mkdir(exist_ok=False, parents=True)
    snapshot(log)
    if args.seed is not None:
        ppo.seed = args.seed
    torch.manual_seed(ppo.seed)
    env, _ = task_registry.make_env(TASK, env_cfg=cfg, train_cfg=ppo,
                                    num_envs=args.num_envs, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, TASK, train_cfg=ppo, log_dir=log,
                                                 resume=args.resume, device=args.device)
        result = runner.learn_until(args.deadline or float('inf'),
            None if args.iterations is None else runner.iteration + args.iterations)
        print(json.dumps(result, indent=2))
    finally:
        env.close()
