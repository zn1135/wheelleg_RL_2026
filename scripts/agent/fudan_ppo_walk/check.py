"""No stepping or optimizer updates: reference, geometry and interface preparation gate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
from ct_mjlab.fudan_ppo_walk.contract import CONTRACT,make_network,PPO_CONFIG,WEIGHTS
from ct_mjlab.rsl_rl.algorithms.fudan_ppo import PPO
from ct_mjlab.fudan_ppo_walk.protocol import manifest
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import make_env_cfg
from ct_mjlab.envs.chuanliantui_stand.mdp import standing_pose


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(791)
    policy=make_network();algorithm=PPO(policy,**PPO_CONFIG)
    obs=torch.randn(8,25);history=torch.randn(8,125)
    policy.update_distribution(obs,history)
    policy.action_mean.sum().backward()
    assert all(p.grad is None for p in policy.encoder.parameters())
    assert any(p.grad is not None for p in policy.actor.parameters())
    primary={id(p) for g in algorithm.optimizer.param_groups for p in g['params']}
    extra={id(p) for g in algorithm.extra_optimizer.param_groups for p in g['params']}
    assert primary.isdisjoint(extra)
    assert extra=={id(p) for p in policy.encoder.parameters()}
    assert policy.critic[0].in_features==144 and policy.actor[0].in_features==28
    cfg=make_env_cfg()
    assert not cfg.commands and not cfg.rewards and not cfg.terminations
    assert set(cfg.events)=={'reset_pose','fudan_randomization'}
    assert cfg.decimation==5 and cfg.sim.mujoco.timestep==.002
    assert cfg.actions['pd'].action_delay_s==0. and not cfg.actions['pd'].randomize_delays
    protocol=manifest();assert len(protocol['samples'])==987
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    root=Path(__file__).resolve().parents[3]
    reference=root/'artifacts/fudan_ppo_walk/fudan-ppo-walk-20261101/reference'
    hashes=json.loads((reference/'manifest.json').read_text())
    assert all(hashlib.sha256((reference/path).read_bytes()).hexdigest()==sha for path,sha in hashes.items())
    package=root/'mjlab_training/src/ct_mjlab/fudan_ppo_walk'
    for file in package.glob('*.py'):ast.parse(file.read_text())
    geometry={str(h):standing_pose(h) for h in (.22,.26,.30)}
    for value in geometry.values():assert value['closed_chain_residual_m']<1e-8
    (args.output/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    report=dict(contract=CONTRACT,static_pass=True,physics_steps=0,optimizer_updates=0,
        actor=25,history=125,environment_critic=141,network_critic=144,action=6,
        optimizer_parameter_disjoint=True,actor_gradient_to_encoder=False,
        reference_files=len(hashes),reward_terms=len(WEIGHTS),formal_samples_per_engine=987,
        protocol_sha256=protocol['sha256'],limits='No physical or behavioral validation')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
