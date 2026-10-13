#!/usr/bin/env python3
"""Check full r11 -> r13 PPO restoration and finite steps; no learning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import torch
from ct_mjlab.reference_training.entry import (
    RecoverVecEnv, runner_cfg, runner_class, validate_wide_height_gate_resume,
)

def equal_tree(left, right):
    if torch.is_tensor(left):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_tree(a,b) for a,b in zip(left,right))
    return left == right

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    validate_wide_height_gate_resume(source)
    for field in ('asset_sha256','reset_bank_sha256','ppo'):
        bad=copy.deepcopy(source);bad['infos']['interface'][field]='incompatible'
        try:validate_wide_height_gate_resume(bad)
        except ValueError:pass
        else:raise AssertionError(f'incompatible {field} accepted')
    bad=copy.deepcopy(source);bad.pop('optimizer_state_dict')
    try:validate_wide_height_gate_resume(bad)
    except ValueError:pass
    else:raise AssertionError('weight-only migration accepted as full resume')
    env=RecoverVecEnv(num_envs=args.num_envs,device='cuda:0',randomize=True,seed=42)
    try:
        runner=runner_class()(env,runner_cfg(),None,device='cuda:0')
        runner.resume_with_wide_height_gate(str(args.checkpoint))
        restored=runner.alg.save()
        for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):
            assert equal_tree(source[key],restored[key]),key
        assert runner.current_learning_iteration==source['iter']+1
        assert runner._effective_training_iteration()==source['infos']['huananhu_training_iteration']
        assert runner.alg.learning_rate==source['optimizer_state_dict']['param_groups'][0]['lr']
        with tempfile.TemporaryDirectory(prefix='ct-r13-resume-') as temp:
            output=Path(temp)/'resume_state.pt';runner.save(str(output))
            saved=torch.load(output,map_location='cpu',weights_only=False)
            assert saved['infos']['interface']['training_rewards']['height_gate']==[.05,.10]
            assert saved['infos']['interface']['training_rewards']['knee_limit_weight']==0.
            assert len(saved['infos']['reward_migration_history'][-1]['changes'])==2
            runner.load(str(output))
            assert equal_tree(restored,runner.alg.save())
        for _ in range(16):
            obs,reward,done,extra=env.step(torch.zeros(args.num_envs,6,device='cuda:0'))
            assert all(torch.isfinite(obs[k]).all() for k in ('actor','critic'))
            assert torch.isfinite(reward).all()
            assert env.unwrapped._huananhu_rewards.height_gate==(.05,.10)
            assert not env.unwrapped.command_manager.get_term('standup').numerical_failure.any()
        result=dict(source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            actor_exact=True,critic_exact=True,adam_exact=True,roundtrip_exact=True,
            next_iteration=runner.current_learning_iteration,
            source_completed_updates=source['infos']['huananhu_training_iteration'],
            learning_rate=runner.alg.learning_rate,migration=runner.reward_migration_history[-1],
            invalid_contracts_rejected=True,missing_optimizer_rejected=True,
            gpu_num_envs=args.num_envs,gpu_control_steps=16,numerical_failure=0,
            scope='full training-state restoration and finite steps; no PPO updates')
        with args.out.open('x') as out:
            json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        env.close()

if __name__=='__main__':main()
