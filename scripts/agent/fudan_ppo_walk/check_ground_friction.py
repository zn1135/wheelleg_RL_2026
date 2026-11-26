"""Check direct/legacy friction fields and generated Warp GPU contacts; no learning."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg


def material_record(wrapper):
    env=wrapper.env;model=env.sim.mj_model
    bodies=set(wrapper.robot.indexing.body_ids.cpu().tolist())
    geoms=[i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies]
    terrain=model.geom('terrain').id
    parameter=env._fudan_parameters['friction']
    direct=wrapper.cfg.domain_rand.ground_friction_range is not None
    coefficient=parameter if direct else (parameter+.5)/2
    assert torch.allclose(env.sim.model.geom_friction[:,geoms,0],coefficient[:,None].expand(-1,len(geoms)))
    floor=env.sim.model.geom_friction[:,terrain,0]
    expected_floor=parameter if direct else torch.full_like(parameter,float(model.geom_friction[terrain,0]))
    assert torch.allclose(floor,expected_floor)
    # 自转和滚动摩擦保持原值；本次只随机滑动摩擦。
    other=env.sim.get_default_field('geom_friction')[geoms,1:]
    assert torch.allclose(env.sim.model.geom_friction[:,geoms,1:],other[None].expand(wrapper.num_envs,-1,-1))
    return dict(mode='direct_ground_contact' if direct else 'legacy_average',
        contact_coefficient_minmax=[float(coefficient.min()),float(coefficient.max())],
        ground_coefficient_minmax=[float(floor.min()),float(floor.max())],
        sampled_coefficient=coefficient.cpu().tolist(),robot_priority=np.unique(model.geom_priority[geoms]).tolist(),
        terrain_priority=int(model.geom_priority[terrain]))


def contact_record(wrapper):
    sim=wrapper.env.sim;data=sim.wp_data;model=sim.mj_model
    count=int(data.nacon.numpy()[0]);pairs=data.contact.geom.numpy()[:count]
    worlds=data.contact.worldid.numpy()[:count];actual=data.contact.friction.numpy()[:count,:2]
    floor=model.geom('terrain').id
    wheel_bodies={wrapper.robot.indexing.body_ids[wrapper.robot.body_names.index(name)].item()
                  for name in ('lfwheel','rfwheel')}
    mask=(pairs==floor).any(axis=1)
    wheels=mask & np.isin(model.geom_bodyid[np.maximum(pairs,0)],list(wheel_bodies)).any(axis=1)
    coefficient=np.asarray(material_record(wrapper)['sampled_coefficient'])
    expected=np.repeat(coefficient[worlds[mask],None],2,axis=1)
    assert np.allclose(actual[mask],expected,atol=1e-7,rtol=0)
    return dict(ground_contacts=int(mask.sum()),wheel_ground_contacts=int(wheels.sum()),
        worlds_with_wheel_contacts=np.unique(worlds[wheels]).tolist(),
        maximum_coefficient_error=float(np.max(np.abs(actual[mask]-expected))) if mask.any() else 0.)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    reports={}
    for name,bounds in [('legacy',None),('low',[.4,.4]),('high',[1.2,1.2]),('random',[.4,1.2])]:
        with ExitStack() as stack:
            stack.enter_context(patch.object(FudanWalkCfg.domain_rand,'ground_friction_range',bounds))
            for method in ('torch.optim.Adam.step','torch.nn.Linear.forward'):
                stack.enter_context(patch(method,side_effect=AssertionError('No network forward or learning')))
            wrapper=FudanEnv(20,seed=30001,randomized=True)
            try:
                before=material_record(wrapper)
                if bounds is not None:
                    values=np.asarray(before['sampled_coefficient'])
                    assert values.min()>=bounds[0]-1e-7 and values.max()<=bounds[1]+1e-7
                    if name=='random':assert values.min()<.5 and values.max()>1.1
                wrapper.reset_idx(torch.tensor([1,4,9],device=wrapper.device))
                assert material_record(wrapper)==before
                contacts=[]
                for _ in range(40):
                    wrapper.step(torch.zeros(20,6,device=wrapper.device))
                    contacts.append(contact_record(wrapper))
                assert sum(row['wheel_ground_contacts'] for row in contacts)>0
                assert torch.isfinite(wrapper.env.sim.data.qpos).all() and torch.isfinite(wrapper.env.sim.data.qvel).all()
                reports[name]=dict(material=before,wheel_contact_observations=sum(row['wheel_ground_contacts'] for row in contacts),
                    maximum_contact_coefficient_error=max(row['maximum_coefficient_error'] for row in contacts),
                    physics_steps=40*wrapper.cfg.control.decimation,partial_reset_preserves_material=True)
            finally:wrapper.env.close()
    report=dict(passed=True,cases=reports,optimizer_updates=0,model_forwards=0,backend='mujoco_warp_gpu',
        scope='Friction endpoints, per-world sampling, actual generated contacts and legacy mapping; not behavior acceptance')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
