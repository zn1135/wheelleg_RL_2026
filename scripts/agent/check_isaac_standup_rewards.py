#!/usr/bin/env python3
"""Source-formula parity and finite GPU reward/reset/entry checks; no training."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.isaac_standup_rewards import WEIGHTS, TIGHT_WEIGHTS, RewardState, reward_values, velocity_scores
from ct_mjlab.smp.ppo import configure_smp, checkpoint_bc_finetune, experiment_profile
from ct_mjlab.smp.finetune import PROFILE_ISAAC, RECIPE_ISAAC, PROFILE_ISAAC_V2, RECIPE_ISAAC_V2
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT/'wheel_legged_gym/envs/base/legged_robot.py',
           ROOT/'wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup.py',
           ROOT/'wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup_config.py']


def source_class(path, name, base, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    # 只执行原文件中纯张量奖励方法，不导入Isaac运行时或混用Python环境。
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
               (n.name.startswith('_reward_') or n.name.startswith('_base_link_') or
                n.name in ('_has_wheel_contact', '_both_wheels_airborne'))]
    cls.body = methods
    cls.bases = [ast.Name(id=base, ctx=ast.Load())] if base else []
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def parity():
    torch.manual_seed(42)
    n = 128
    ns = {'torch': torch}
    source_class(SOURCES[0], 'LeggedRobot', None, ns)
    cls = source_class(SOURCES[1], 'ChuanliantuiStandup', 'LeggedRobot', ns)
    source = cls()
    cfg = ast.parse(SOURCES[2].read_text())
    parent = next(x for x in cfg.body if isinstance(x, ast.ClassDef) and x.name == 'ChuanliantuiStandupCfg')
    rewards = next(x for x in parent.body if isinstance(x, ast.ClassDef) and x.name == 'rewards')
    scales = next(x for x in rewards.body if isinstance(x, ast.ClassDef) and x.name == 'scales')
    weights = {x.targets[0].id: ast.literal_eval(x.value) for x in scales.body if isinstance(x, ast.Assign)}
    assert {k: v for k, v in weights.items() if v} == WEIGHTS
    source.cfg = NS(rewards=NS(tracking_sigma=.25, height_reward_sigma=.01,
        height_reward_contact_factor=.2, base_link_reward_force_scale=5.,
        recovered_reward_duration_s=.5, knee_limit_margin_rad=.05),
        standup=NS(initial_base_height=.15, wheel_contact_force_threshold=1.))
    limits = torch.tensor([[-.5, .8]]*4)
    knees = torch.tensor([[-.12, .77], [-.77, .12]])
    q, dq = torch.randn(n, 4), torch.randn(n, 4)*300
    kq = q[:, (1, 3)]
    inp = dict(command=torch.cat((torch.randn(n, 2), torch.full((n, 1), .33648)), -1),
        height=torch.rand(n)*.5, gravity=torch.randn(n, 3),
        linear_velocity=torch.randn(n, 3)*3, angular_velocity=torch.randn(n, 3)*4,
        base_force=torch.rand(n)*10, wheel_supported=torch.rand(n, 2)>.5,
        landed=torch.rand(n)>.5, stood=torch.rand(n)>.5, standing_time=torch.rand(n),
        leg_swing=torch.randn(n, 2), leg_q=q, leg_dq=dq, soft_limits=limits,
        knee_q=kq, knee_limits=knees, torque=torch.randn(n, 6)*80,
        action=torch.randn(n, 6)*10, previous_action=torch.randn(n, 6),
        previous_previous_action=torch.randn(n, 6))
    source.commands=inp['command']; source.base_height=inp['height']
    source.projected_gravity=inp['gravity']; source.base_lin_vel=inp['linear_velocity']
    source.base_ang_vel=inp['angular_velocity']; source.has_stood=inp['stood']
    source.has_landed=inp['landed']; source.standing_time=inp['standing_time']
    source.theta0=inp['leg_swing']; source.standup_curriculum_unlocked=False
    source.contact_forces=torch.zeros(n, 3, 3)
    source.contact_forces[:, 0, 2]=inp['base_force']
    source.contact_forces[:, 1:, 2]=inp['wheel_supported']*2.
    source.feet_indices=torch.tensor([1, 2]); source.termination_contact_indices=torch.tensor([0])
    source.dof_pos=q[:, (0, 1, 0, 2, 3, 0)]; source.dof_vel=dq[:, (0, 1, 0, 2, 3, 0)]
    source.dof_pos_limits=limits[[0, 1, 0, 2, 3, 0]]
    source.knee_dof_indices=torch.tensor([1, 4]); source.knee_hard_limits=knees
    source.torques=inp['torque']; source.actions=inp['action']
    source.last_actions=torch.stack((inp['previous_action'], inp['previous_previous_action']), -1)
    source.reward_scales=WEIGHTS
    values=reward_values(**inp)
    errors={}
    for name, weight in WEIGHTS.items():
        reference=getattr(source, '_reward_'+name)()
        torch.testing.assert_close(values[name], reference)
        expected=(reference*weight*.01).clamp(-.01, .01)
        actual=values[name].clamp(-1/abs(weight), 1/abs(weight))*weight*.01
        torch.testing.assert_close(actual, expected)
        errors[name]=float((actual-expected).abs().max())
    return errors


def check(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    errors=parity()
    tight = args.variant == 'isaac-standup-v2'
    recipe = RECIPE_ISAAC_V2 if tight else RECIPE_ISAAC
    profile = PROFILE_ISAAC_V2 if tight else PROFILE_ISAAC
    speed_check = {}
    if tight:
        speeds=torch.tensor([0., .002, .005, .01, .02, .05, .1, .2], requires_grad=True)
        v=torch.stack((speeds, speeds*0., speeds*0.), -1)
        command=torch.zeros_like(v)
        a,b=velocity_scores(command,v,torch.ones(8,dtype=torch.bool),tight=True)
        score=a+.2*b
        assert (score[:-1]>score[1:]).all()
        score.sum().backward(); assert (speeds.grad[1:] < 0).all()
        for kwargs in (dict(tight=True), dict(tight=False)):
            off=velocity_scores(command,v,torch.zeros(8,dtype=torch.bool),**kwargs)
            assert not off[0].any() and not off[1].any()
        side=velocity_scores(command,v[:, [1,0,2]],torch.ones(8,dtype=torch.bool),tight=True)
        torch.testing.assert_close(side[0]+.2*side[1],score)
        import numpy as np
        with np.load(args.trace) as trace:
            velocity=torch.from_numpy(trace['features'][1500:,:,4:7].copy()).reshape(-1,3)
        old=velocity_scores(torch.zeros_like(velocity),velocity,torch.ones(len(velocity),dtype=torch.bool))
        new=velocity_scores(torch.zeros_like(velocity),velocity,torch.ones(len(velocity),dtype=torch.bool),tight=True)
        speed_check=dict(speeds=speeds.detach().tolist(),weighted_scores=score.detach().tolist(),
            old_trace_score=float((.2*old[0]+.2*old[1]).mean()),old_peak=.2,
            new_trace_score=float((new[0]+.2*new[1]).mean()),new_peak=1.,
            interpretation='Re-score stored late velocities with stood=True; not a new policy rollout or learning result.')
    from dataclasses import asdict
    cfg, train=task_registry.get_cfgs('chuanliantui')
    candidate_cfg, candidate_train=copy.deepcopy((cfg, train))
    configure_smp(cfg, train, args.prior, policy_contract=HANDOVER_OBSERVATION_CONTRACT, bc_finetune='v2')
    old=copy.deepcopy(cfg)
    cfg, train=candidate_cfg, candidate_train
    configure_smp(cfg, train, args.prior, policy_contract=HANDOVER_OBSERVATION_CONTRACT, bc_finetune=args.variant)
    assert set(cfg.rewards)=={'smp_motion', *WEIGHTS}
    assert experiment_profile(cfg)==profile
    assert {k:v.weight for k,v in cfg.rewards.items() if k!='smp_motion'} == (TIGHT_WEIGHTS if tight else WEIGHTS)
    # 奖励以外的环境字段与v2逐项一致，包含初态、命令、观测和时序。
    old_dict, new_dict=asdict(old), asdict(cfg)
    old_dict.pop('rewards'); new_dict.pop('rewards')
    assert old_dict==new_dict, {key: (old_dict[key], new_dict[key]) for key in old_dict if old_dict[key] != new_dict[key]}
    cfg.seed=train.seed=42
    env, _=task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=20)
    try:
        raw=env.unwrapped
        runner, _=task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        if args.ppo_init:
            critic_before=copy.deepcopy(runner.alg.save()['critic_state_dict'])
            runner.initialize_from_ppo(args.ppo_init)
            source=torch.load(args.ppo_init,map_location='cpu',weights_only=False)
            for k,v in source['actor_state_dict'].items():
                torch.testing.assert_close(runner.alg.save()['actor_state_dict'][k].cpu(),v,atol=0,rtol=0)
            for k,v in critic_before.items():
                torch.testing.assert_close(runner.alg.save()['critic_state_dict'][k],v,atol=0,rtol=0)
            assert runner.alg.completed_iterations==0 and not runner.alg.optimizer.state
            for k,v in runner.alg.reference.state_dict().items():
                torch.testing.assert_close(v.cpu(),source['actor_state_dict']['mlp.'+k],atol=0,rtol=0)
        else:
            runner.initialize_from_bc(args.bc)
        before=copy.deepcopy(runner.alg.save())
        legacy = RewardState(raw)
        original_compute = RewardState.compute
        def compare_compute(state, env):
            values = original_compute(state, env)
            if state is not legacy:
                # 在奖励实际执行的同一拍比较，不能等step结束后动作历史已滚动再重算。
                previous = original_compute(legacy, env)
                for name in WEIGHTS:
                    if name not in ('tracking_lin_vel', 'tracking_lin_vel_enhance'):
                        torch.testing.assert_close(values[name],previous[name],atol=0,rtol=0,msg=name)
            return values
        with torch.inference_mode(), patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training')), \
                patch.object(RewardState, 'compute', compare_compute):
            obs=env.get_observations()
            for tick in range(300):
                obs, total, done, extras=env.step(runner.alg.actor(obs))
                assert obs['actor'].shape==(20, 39) and obs['critic'].shape==(20, 82)
                assert torch.isfinite(total).all()
                torch.testing.assert_close(raw.reward_manager._step_reward.sum(-1)*raw.step_dt, total)
                state=raw._isaac_standup_rewards
                count=state.standing_time.clone()
                state.compute(raw)
                torch.testing.assert_close(state.standing_time, count, atol=0, rtol=0)
                if tick < 20: assert not raw.action_manager.get_term('pd').torques.any()
                if tick==149:
                    preserved=(state.landed[5:].clone(), state.stood[5:].clone(), state.standing_time[5:].clone())
                    raw.reset(env_ids=torch.arange(5, device=raw.device))
                    legacy.reset(torch.arange(5,device=raw.device))
                    assert not state.landed[:5].any() and not state.stood[:5].any() and not state.standing_time[:5].any()
                    for actual, expected in zip((state.landed[5:], state.stood[5:], state.standing_time[5:]), preserved):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    obs=raw.observation_manager.compute(update_history=False)
            stood=int(state.stood.sum())
            path=args.output/'untrained-interface.pt'
            runner.save(str(path))
            assert checkpoint_bc_finetune(path)==args.variant
            resumed, _=task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
            resumed.load(str(path))
            assert resumed.alg.recipe==recipe and not resumed.alg.optimizer.state
            if args.ppo_init:
                assert resumed.ppo_init==runner.ppo_init
                try: resumed.load(str(args.ppo_init))
                except ValueError: pass
                else: raise AssertionError('Cross-reward full resume must be rejected')
            for name in ('actor_state_dict', 'critic_state_dict'):
                for key, value in before[name].items():
                    torch.testing.assert_close(resumed.alg.save()[name][key], value, atol=0, rtol=0)
        result=dict(passed=True, optimizer_updates=0, formula_max_abs_error=errors,
            recipe=recipe, speed_check=speed_check, old_other_17_rewards_unchanged=True,
            ppo_actor_transfer=bool(args.ppo_init), fresh_critic_and_optimizer=True,
            num_envs=20, steps=300, reward_stood_at_end=stood, partial_reset_isolation=True,
            reward_dt_once=True, reward_history_once_per_step=True, only_rewards_differ_from_v2=True,
            checkpoint_roundtrip=True, sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
            scope='19 source reward formula/weight/clip checks, 20-env finite GPU replay and partial reset. No training or behavior acceptance.')
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally: env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bc', 'prior', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--variant',choices=('isaac-standup','isaac-standup-v2'),default='isaac-standup')
    p.add_argument('--ppo-init',type=Path)
    p.add_argument('--trace',type=Path)
    check(p.parse_args())
