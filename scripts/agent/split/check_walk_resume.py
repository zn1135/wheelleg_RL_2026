#!/usr/bin/env python3
"""Check the real walk checkpoint, curriculum boundaries and finite training rollout; never learn."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split.runner import validate
from scripts.agent.split.check_chuanliantui_two_policy import digest, tensors_equal
from scripts.agent.split.check_walk_direction_curriculum import sampling_checks, accounting_checks


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,train=task_registry.get_cfgs('chuanliantui_walk')
    cfg.seed=train.seed=42
    env,_=task_registry.make_env('chuanliantui_walk',env_cfg=cfg,train_cfg=train,
        num_envs=20,device=args.device)
    raw=env.unwrapped
    try:
        sampling=sampling_checks(raw)
        accounting=accounting_checks(raw)
        raw.reset(seed=42)
        built,_=task_registry.make_alg_runner(env,'chuanliantui_walk',train_cfg=train,
            resume=args.checkpoint,device=args.device)
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        state=raw.command_manager.get_term('standup')
        validate(saved['infos'],state.cfg)
        assert built.current_learning_iteration==saved['iter']+1
        assert raw.common_step_counter==saved['infos']['env_state']['common_step_counter']
        assert tensors_equal(state.checkpoint_state(),saved['infos']['standup_curriculum'])
        for key,actual in (
                ('actor_state_dict',built.alg.actor.state_dict()),
                ('critic_state_dict',built.alg.critic.state_dict()),
                ('optimizer_state_dict',built.alg.optimizer.state_dict()),
                ('encoder_optimizer_state_dict',built.alg.encoder_optimizer.state_dict())):
            assert tensors_equal(actual,saved[key]),key
        assert built.alg.learning_rate==built.alg.optimizer.param_groups[0]['lr']
        restored=copy.deepcopy(state.checkpoint_state())
        policy=built.get_inference_policy(device=args.device)
        obs=raw.get_observations()
        nonzero_vx=0; failures=0
        for _ in range(256):
            action=policy(obs,stochastic_output=True)
            if train.clip_actions is not None:action=action.clamp(-train.clip_actions,train.clip_actions)
            obs,reward,terminated,timeout,_=raw.step(action)
            assert all(torch.isfinite(obs[name]).all() for name in ('actor','critic','history'))
            assert torch.isfinite(action).all() and torch.isfinite(reward).all()
            nonzero_vx+=int((state.command[:,0].abs()>.01).sum())
            failures+=int(terminated.sum())
        for key,actual in (
                ('actor_state_dict',built.alg.actor.state_dict()),
                ('critic_state_dict',built.alg.critic.state_dict()),
                ('optimizer_state_dict',built.alg.optimizer.state_dict()),
                ('encoder_optimizer_state_dict',built.alg.encoder_optimizer.state_dict())):
            assert tensors_equal(actual,saved[key]),key
        report=dict(verified=True,checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest(args.checkpoint),
            source_iteration=saved['iter'],next_iteration=built.current_learning_iteration,
            actor_critic_and_two_optimizers_exact=True,adaptive_learning_rate_restored=built.alg.learning_rate,
            restored_curriculum=restored,source_optimizer_steps={key:sorted({int(v['step']) for v in saved[key]['state'].values()})
                for key in ('optimizer_state_dict','encoder_optimizer_state_dict')},
            sampling=sampling,accounting=accounting,training_configuration=True,num_envs=20,steps=256,
            nonzero_vx_samples=nonzero_vx,finite_rollout_terminations=failures,parameter_updates=0,
            scope='Native full checkpoint restoration, direction boundaries and finite rollout; not behavior acceptance')
        (args.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(verified=True,next_iteration=built.current_learning_iteration,parameter_updates=0)))
    finally:
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Preflight cannot train')):
        main(parser.parse_args())
