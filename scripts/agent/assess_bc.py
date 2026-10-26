#!/usr/bin/env python3
"""Audit the authorized BC budget and heldout prediction errors; never update."""
import argparse,hashlib,json
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ct_mjlab.bc import ActionDataset,load_actor,LEGACY_SAMPLING


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--corrections',type=Path,action='append',default=[]);p.add_argument('--expected-updates',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);actor,saved=load_actor(a.checkpoint)
 data=ActionDataset([a.dataset],a.corrections,action_sampling=saved['dataset'].get('action_sampling',LEGACY_SAMPLING))
 assert saved['iteration']==a.expected_updates and saved['dataset']['fingerprint']==data.description['fingerprint']
 if saved.get('initialization'):
  assert saved['total_actor_updates']==saved['initialization']['updates']+a.expected_updates
 steps=sorted({int(v['step']) for v in saved['optimizer_state_dict']['state'].values()});assert steps==[a.expected_updates]
 events=EventAccumulator(str(a.checkpoint.parent),size_guidance={'scalars':0}).Reload()
 assert [v.step for v in events.Scalars('BC/train_mse_scaled')]==list(range(1,a.expected_updates+1))
 assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
 obs,target=data.sample('validation',2048,torch.Generator().manual_seed(714));err=actor.mlp(obs)-target
 assert torch.isfinite(err).all()
 result=dict(passed=True,optimizer_updates_during_assessment=0,authorized_updates=a.expected_updates,
  total_actor_updates=saved.get('total_actor_updates',saved['iteration']),initialization=saved.get('initialization'),
  optimizer_steps=steps,training_events=a.expected_updates,dataset_fingerprint=data.description['fingerprint'],
  validation_scaled_mse=float((err/saved['target_scale'].cpu()).square().mean()),
  validation_action_mae=err.abs().mean(0).tolist(),validation_action_rmse=err.square().mean(0).sqrt().tolist(),
  action_std=actor.distribution.std_param.tolist(),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
  scope='Saved budget, network values and heldout paired prediction; independent closed loop assessed separately.')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
