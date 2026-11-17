"""Capture first real Warp ray miss under stochastic actions; no optimizer updates."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
from ct_mjlab.fudan_ppo_walk.policy import load_policy


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--restore-rng',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=dict(optimizer_updates=0,ray_miss_reproduced=False,steps=0);env=None
    torch.set_num_threads(1);torch.manual_seed(1)
    try:
        with patch('torch.optim.Adam.step',side_effect=AssertionError('No updates')):
            policy,saved=load_policy(a.checkpoint,device='cuda:0')
            env=StepEnv(a.num_envs,seed=1)
            if len(saved.get('terrain_levels',[]))==a.num_envs:
                env.terrain.terrain_levels.copy_(saved['terrain_levels'].to(env.device))
                env.terrain.terrain_types.copy_(saved['terrain_types'].to(env.device))
            if a.restore_rng:
                torch.set_rng_state(saved['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
            original=env._update_state
            def update(ids=None):
                try:return original(ids)
                except FloatingPointError:
                    scan=env.env.scene['step_scan'];support=env.env.scene['step_support']
                    bad=(scan.data.distances<0).any(-1)|(support.data.distances<0).any(-1)
                    q=env.env.sim.data.qpos;origin=env.env.scene.env_origins
                    relative=q[:,:3]-origin
                    rows=[]
                    for i in bad.nonzero().flatten().tolist():
                        rows.append(dict(env=i,mode=int(env.modes[i]),level=int(env.terrain.terrain_levels[i]),
                            relative_root=relative[i].cpu().tolist(),gravity=env.gravity[i].cpu().tolist(),
                            scan_misses=int((scan.data.distances[i]<0).sum()),
                            support_misses=int((support.data.distances[i]<0).sum()),
                            scan_origins_min=scan._cached_world_origins[i].amin(0).cpu().tolist(),
                            scan_origins_max=scan._cached_world_origins[i].amax(0).cpu().tolist()))
                    np.savez_compressed(a.output/'failure.npz',qpos=q.cpu().numpy(),qvel=env.env.sim.data.qvel.cpu().numpy(),
                        origin=origin.cpu().numpy(),scan_distance=scan.data.distances.cpu().numpy(),
                        scan_origins=scan._cached_world_origins.cpu().numpy(),
                        support_distance=support.data.distances.cpu().numpy())
                    report.update(ray_miss_reproduced=True,affected=rows)
                    report.update(constraint_peak=int(env.contacts.constraint_peak),
                        capacity_overflow=bool(env.contacts.capacity_overflow.any()))
                    raise
            env._update_state=update
            obs,_,history=env.reset()
            with torch.inference_mode():
                for i in range(a.steps):
                    if time.time()>=a.deadline:raise TimeoutError('Diagnostic deadline')
                    report['steps']=i
                    obs,_,_,_,_,history=env.step(policy.act(obs,history))
                    if (i+1)%200==0:print(json.dumps(dict(completed_steps=i+1,optimizer_updates=0)),flush=True)
    except FloatingPointError as exc:report['captured_error']=str(exc)
    finally:
        if env is not None:env.close()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
