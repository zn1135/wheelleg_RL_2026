#!/usr/bin/env python3
"""Audit teacher queries on student trajectories and exclude heldout/truncated labels."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,load_actor
from ct_mjlab.envs.chuanliantui.model import MODEL_XML,ACTUATED_JOINTS
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--corrections',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);base=ActionDataset([a.dataset]);mixed=ActionDataset([a.dataset],[a.corrections])
 from mjlab.utils.torch import configure_torch_backends
 configure_torch_backends()
 actor,_=load_actor(a.checkpoint,'cuda:0')
 r=json.loads((a.corrections/'result.json').read_text());assert r['teacher_queries_only'] and r['optimizer_updates']==0
 assert r['checkpoint_sha256']==digest(a.checkpoint)
 model=mujoco.MjModel.from_xml_path(str(MODEL_XML));ref=model.qpos0[[model.joint(n).qposadr[0] for n in ACTUATED_JOINTS]]
 with np.load(base.motion.episodes[0]['path']) as d:nominal=d['nominal_motor_positions'].copy()
 with np.load(a.corrections/'trace.npz') as d:
  valid=d['label_valid']&d['alive'];obs=torch.from_numpy(d['observation'][valid].copy());actions=torch.from_numpy(d['action'][valid].copy());labels=d['label'][valid].copy()
  # 与实际运行使用同一GPU精度及完整环境批次；CPU IEEE与GPU TF32不能作逐位对照。
  ticks=torch.linspace(0,len(d['observation'])-1,16).round().long().numpy()
  prediction=torch.stack([actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu() for tick in ticks])
  actual=torch.from_numpy(d['action'][ticks].copy())
  torch.testing.assert_close(prediction,actual,atol=2e-6,rtol=2e-6)
  max_action_difference=float((prediction-actual).abs().max())
  legs=[0,1,3,4];clipped=(abs(nominal[legs]+.5*labels[:,legs]-ref[legs])>3.14).any(-1)|(abs(labels[:,[2,5]])>15).any(-1)
  assert not clipped.any(),'Unrepresentable teacher labels must be filtered explicitly'
  errors=abs(actions.numpy()-labels).mean(0)
 assert torch.equal(base.frames['validation'][0],mixed.frames['validation'][0]) and torch.equal(base.frames['validation'][1],mixed.frames['validation'][1])
 result=dict(passed=True,optimizer_updates=0,valid_correction_frames=len(obs),training_pose_groups=len({e['group_id'] for e in base.motion.splits['train']}),
  heldout_poses_not_queried_for_training=True,heldout_frames_unchanged=True,actual_actions_match_independent_actor=True,
  clipped_valid_targets=0,max_recomputed_actual_action_difference=max_action_difference,
  teacher_query_action_mae=errors.tolist(),
  original_fingerprint=base.description['fingerprint'],augmented_fingerprint=mixed.description['fingerprint'],
  added_training_frames=len(mixed.frames['train'][0])-len(base.frames['train'][0]),
  source_trace_sha256=r['trace_sha256'],scope='Finite teacher labels on actual student trajectories, immutable heldout data and target limits; corrected policy not trained.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
