"""本次授权：扫描5838续训1000轮到6838，坡5～20°、实际摩擦固定0.8。"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

from . import resume_regions3000 as regional
from .resume_pd200_random_from3000 import ROOT,PYTHON,XML,sha,write
from .run_torque_ablation import run_command

MODULE='scripts.agent.fudan_ppo_walk.resume_slope5to20_mu08_1000'
SOURCE_RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261116_critic_scan_regions_resume5000_to6000_01'
SOURCE=SOURCE_RUN/'stage_5838.pt'
SOURCE_SHA='96bcc67c3eb9453ff527e3bb33dac57c940e8c05a911760f8b63c9fb2ae77444'
RECIPE=ROOT/'artifacts/fudan_ppo_walk/20261116-regional-slope5to20-friction08-01'
PREVIOUS=ROOT/'artifacts/fudan_ppo_walk/20261116-critic-scan-resume5000-6000-01'
LOG=ROOT/'logs/mjlab/fudan_ppo_walk/20261116_scan_slope5to20_mu08_resume5838_to6838_01'


def prepare(out):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from ct_mjlab.envs.fudan_ppo_walk.terrain import record,critic_record
    assert not LOG.exists() and not out.exists()
    assert sha(SOURCE)==SOURCE_SHA
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    start=source['iterations'];assert start==5838 and not source['partial_update']
    assert source['num_envs']==4096 and source['transitions']==start*4096*48+184320
    assert source['primary_updates']==source['encoder_updates']==start*20
    assert all(torch.isfinite(v).all() for v in source['model_state_dict'].values())
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in source[key]['state'].values()}=={start*20}
        assert all(torch.isfinite(v).all() for state in source[key]['state'].values()
                   for v in state.values() if isinstance(v,torch.Tensor))
    previous=json.loads((PREVIOUS/'budget.json').read_text())
    config=json.loads((RECIPE/'target-effective-config.json').read_text())
    before=previous['actual_config']
    differences={f'{s}.{k}':[before[s][k],config[s][k]] for s in before for k in before[s]
                 if before[s][k]!=config[s][k]}
    assert differences=={'terrain.slope_degrees':[[10.,20.],[5.,20.]],
                        'domain_rand.ground_friction_range':[[.4,1.2],[.8,.8]]}
    task=FudanWalkCfg()
    for section,values in config.items():
        for key,value in values.items():setattr(getattr(task,section),key,value)
    assert task.terrain.mode=='regions' and task.observations.critic_height_scan
    assert source['critic_observation']==critic_record(task)
    proof=json.loads((RECIPE/'precheck/report.json').read_text())
    assert proof['passed'] and proof['terrain']==record(task) and proof['critic_scan']['passed']
    for section,values in config.items():assert proof['actual_config'][section]==values,section
    # 训练实现相对源快照只允许已确认的默认配置变化；算法和实体控制保持。
    changed=[]
    for name,digest in json.loads((SOURCE_RUN/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/') and sha(ROOT/name)!=digest:changed.append(name)
    assert changed==['mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk_config.py'],changed
    out.mkdir(parents=True)
    budget=dict(authorization='User requested1000 additional training iterations after selecting slopes5-20deg and fixed friction0.8',
        checkpoint=str(SOURCE),checkpoint_sha256=SOURCE_SHA,source_run=str(SOURCE_RUN),
        start_iteration=start,target_iteration=start+1000,additional_iterations=1000,
        num_envs=4096,rollout_steps=48,seed=1,log_dir=str(LOG),actual_config=config,
        terrain=record(task),critic_observation=critic_record(task),runtime_capacity=dict(nconmax=64,njmax=256),
        control_timing=source['control_timing'],disturbances=source['disturbances'],
        command_sampling=source['command_sampling'],height_reward=source['height_reward'],
        physical_wheel_model=source['physical_wheel_model'],xml_sha256=sha(XML),
        inherited_abandoned_rollout_transitions=184320,expected_added_transitions=1000*4096*48,
        expected_added_optimizer_steps=20000,expected_final_optimizer_steps_each=(start+1000)*20,
        only_recipe_changes=differences,initialization='all_networks_both_optimizers_rng_counters_restored_exact_fresh_physics_history',
        previous_training_interruption=str(PREVIOUS/'host-interruption.json'),
        budget_note='New1000-round authorization from5838; no automatic addition of old unfinished budget or unsaved61 rounds',
        prepared_recipe=str(RECIPE/'target-effective-config.json'),created_unix=time.time())
    paths=list((ROOT/'mjlab_training/src/ct_mjlab').rglob('*.py'))
    paths += [Path(__file__),Path(regional.__file__),ROOT/'scripts/agent/fudan_ppo_walk/check_region_terrain.py',
              ROOT/'scripts/agent/fudan_ppo_walk/check_critic_scan.py']
    budget['approved_source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(out/'budget.json',budget);write(out/'target-effective-config.json',config)
    write(out/'source-checkpoint-audit.json',dict(passed=True,sha256=SOURCE_SHA,iteration=start,
        finite_learning_state=True,optimizer_steps_each=start*20,complete_rollout_boundary=True,
        implementation_changed=changed,only_recipe_changes=differences,optimizer_updates=0))
    shutil.copy2(__file__,out/'launcher.py');shutil.copy2(regional.__file__,out/'regional-launcher.py')
    write(out/'source-identity.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=True))
    write(out/'status.json',dict(status='prepared',training_started=False))


def worker(out,stage):
    import numpy as np
    import torch
    from ct_mjlab.envs.fudan_ppo_walk import fudan_ppo_walk as physical
    old_init=physical.FudanEnv.__init__
    def initialized(env,*args,**kwargs):
        old_init(env,*args,**kwargs)
        sim=env.env.sim;values=env.env._fudan_parameters['friction']
        assert torch.allclose(values,torch.full_like(values,.8),atol=1e-7,rtol=0.)
        bodies=env.robot.indexing.body_ids.cpu().numpy()
        geoms=np.union1d(np.flatnonzero(np.isin(sim.mj_model.geom_bodyid,bodies)),
                        env.env._terrain_friction_ids.cpu().numpy()).tolist()
        assert torch.allclose(sim.model.geom_friction[:,geoms,0],torch.full_like(sim.model.geom_friction[:,geoms,0],.8),atol=1e-7,rtol=0.)
        write(out/(stage+'-actual-material.json'),dict(passed=True,num_envs=env.num_envs,
            fixed_sliding_friction_minmax=[float(values.min()),float(values.max())],
            robot_and_terrain_friction_fixed=True,critic_scan=env.cfg.observations.critic_height_scan,
            slope_degrees=env.cfg.terrain.slope_degrees))
    with patch.object(regional,'SOURCE',SOURCE),patch.object(physical.FudanEnv,'__init__',initialized):
        regional.worker(out,stage)


def audit(out):
    with patch.object(regional,'SOURCE',SOURCE):regional.audit(out)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=('prepare','precheck','capacity','train','run','audit'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':return prepare(out)
    if a.mode=='audit':return audit(out)
    if a.mode!='run':return worker(out,a.mode)
    try:
        assert not LOG.exists()
        for stage in ('precheck','capacity','train'):
            run_command(out,stage,[str(PYTHON),'-u','-m',MODULE,stage,'--output',str(out)])
            if stage=='precheck':
                assert json.loads((out/'precheck/report.json').read_text())['critic_scan']['passed']
                assert json.loads((out/'precheck-runtime-audit.json').read_text())['passed']
            elif stage=='capacity':assert json.loads((out/'capacity-audit.json').read_text())['passed']
        audit(out)
        write(out/'status.json',dict(status='training_complete',final_iteration=6838,iterations_added=1000,behavior_accepted=False))
    except BaseException as exc:
        write(out/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
