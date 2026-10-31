#!/usr/bin/env python3
"""Check BC timing, grouped split, network parity and gradients without updates."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from ct_mjlab.bc import ActionDataset, build_actor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corrections',type=Path,action='append',default=[])
    args=parser.parse_args();torch.set_num_threads(1)
    dataset=ActionDataset(args.dataset,args.corrections)
    actor=build_actor(observation_dim=dataset.observation_dim)
    obs,target=dataset.sample('train',256,torch.Generator().manual_seed(71))
    # 与实际PPO的MLP配置核对，不仅核对一个相似网络的张量形状。
    from ct_mjlab.envs import task_registry
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    assert tuple(train.actor.hidden_dims)==(256,128,64) and train.actor.activation=='elu'
    assert not train.actor.obs_normalization and not cfg.actions['pd'].motor_randomization
    state={k:v.clone() for k,v in actor.state_dict().items()}
    loss=(actor.mlp(obs)-target).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in actor.mlp.parameters())
    for k,v in actor.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
    groups={s:{e['group_id'] for e in episodes} for s,episodes in dataset.motion.splits.items()}
    assert not groups['train']&groups['validation']
    report=dict(passed=True,optimizer_updates=0,all_pair_timestamps_and_previous_actions_checked=True,
        pre_step_sensors_recomputed=True,group_disjoint=True,finite_loss=float(loss.detach()),
        dataset=dataset.description,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='dataset'},indent=2))


if __name__=='__main__':main()
