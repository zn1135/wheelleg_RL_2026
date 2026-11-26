"""Low-memory exact GPU replay of contact statistics/episode serialization; no simulator or learning."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import torch
from scripts.agent.fudan_ppo_step.check_speed_optimization import legacy_rows


def load_contacts(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ContactMonitor


def new_rows(path):
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='StepEnv')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='_record_episodes')
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['_record_episodes']


def fixture(n):
    def random(*shape):return torch.randn(*shape,device='cuda:0')
    def sensor(slots):
        return NS(data=NS(force=random(n,slots,3),pos=random(n,slots,3),normal=random(n,slots,3),
                          found=torch.ones(n,slots,device='cuda:0',dtype=torch.int32)))
    scene={'step_base':sensor(1),'step_legs':sensor(11),'step_wheels':sensor(8),
           'robot':NS(indexing=NS(body_ids=torch.arange(16,device='cuda:0'))),
           'step_scan':NS(data=NS(distances=random(n,77)),_cached_world_origins=random(n,77,3)),
           'step_support':NS(data=NS(distances=random(n,2)))}
    scene=type('Scene',(dict,),{})(scene);scene.env_origins=random(n,3)
    physics=NS(num_envs=n,device='cuda:0',physics_dt=.001,scene=scene,
        sim=NS(model=NS(body_mass=torch.ones(n,16,device='cuda:0')),
               data=NS(qvel=random(n,12),qpos=random(n,25))))
    cfg=NS(contact_threshold_N=1.,support_weight_fraction=.1,contact_timeout_s=.2,support_timeout_s=.1)
    return physics,cfg


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline-source',type=Path,required=True)
    p.add_argument('--candidate-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    relative=Path('mjlab_training/src/ct_mjlab/envs/chuanliantui/fudan_ppo_step')
    old_type=load_contacts(args.baseline_source/relative/'contacts.py','old_contact_inputs')
    new_type=load_contacts(args.candidate_source/relative/'contacts.py','new_contact_inputs')
    old_record=legacy_rows(args.baseline_source/relative/'env.py')
    new_record=new_rows(args.candidate_source/relative/'env.py')
    report=dict(passed=False,physics_steps=0,optimizer_updates=0)
    fields=('ticks','count','streak','longest','peak','peak_pos','any_streak','strong_streak','failed',
            'step_contact','supported','side_contact','step_speed_peak','capacity_overflow','constraint_peak','global_contact_peak')
    try:
        torch.set_num_threads(1);torch.manual_seed(1711)
        with torch.inference_mode():
            physics,cfg=fixture(64);a,b=old_type(physics,cfg),new_type(physics,cfg)
            for i in range(500):
                if i%10==0:
                    for monitor in (a,b):
                        monitor.step_contact.zero_();monitor.step_speed_peak.zero_()
                        monitor.step_forces.clear();monitor.step_positions.clear()
                        monitor.record_substeps=i%20==0
                if i%31==0:
                    ids=torch.arange(i%7,64,7,device='cuda:0');a.reset(ids);b.reset(ids)
                physics.scene['step_base'].data.force.uniform_(-20,20)
                physics.scene['step_legs'].data.force.uniform_(-20,20)
                # 精确阈值、相同峰值和运行时质量变化也进入同输入比较。
                physics.scene['step_base'].data.force[0,0]=torch.tensor([1.,0.,0.],device='cuda:0')
                physics.scene['step_legs'].data.force[1]=0.
                physics.scene['step_legs'].data.force[1,:2,0]=2.
                if i==137:physics.sim.model.body_mass.mul_(1.1)
                x,y=a.update(),b.update()
                assert bool(torch.stack([(getattr(a,k)==getattr(b,k)).all() for k in fields]+[(x==y).all()]).all())
                if a.record_substeps:
                    assert torch.equal(a.step_forces[-1],b.step_forces[-1])
                    assert torch.equal(a.step_positions[-1],b.step_positions[-1])
            report.update(exact_contact_frames=500,contact_fields=len(fields),substep_recording_exact=True,
                          threshold_ties_resets_and_mass_change_checked=True)
            physics,cfg=fixture(4096);monitor=new_type(physics,cfg)
            # 记录字段来自同一组GPU张量，比较类型、顺序和缺测诊断，不用模拟回合成功率。
            base=NS(env=physics,contacts=monitor,modes=torch.arange(4096,device='cuda:0')%3,
                terrain=NS(terrain_levels=torch.arange(4096,device='cuda:0')%10),
                success=torch.rand(4096,device='cuda:0')>.5,reverse=torch.rand(4096,device='cuda:0')>.5,
                crossed=torch.rand(4096,device='cuda:0')>.5,ray_failed=torch.zeros(4096,device='cuda:0',dtype=torch.bool),
                max_approach_speed=torch.rand(4096,device='cuda:0'),finished=torch.zeros(4096,device='cuda:0',dtype=torch.bool),
                position=torch.randn(4096,3,device='cuda:0'),gravity=torch.randn(4096,3,device='cuda:0'))
            failure=torch.rand(4096,device='cuda:0')>.5;applied=torch.rand(4096,device='cuda:0')>.5
            monitor.update();episode_cases=[]
            for count in (1,4,64,512):
                ids=torch.arange(count-1,-1,-1,device='cuda:0')
                for missing in (False,True):
                    left,right=copy.copy(base),copy.copy(base)
                    for view in (left,right):
                        view.episode_rows=[];view.finished=base.finished.clone();view.ray_failed=base.ray_failed.clone()
                        if missing:view.ray_failed[ids[:1]]=True
                    old_record(left,ids,failure,applied);new_record(right,ids,failure,applied)
                    assert left.episode_rows==right.episode_rows
                    assert torch.equal(left.finished,right.finished)
                times={}
                for label,function in [('old',old_record),('new',new_record)]:
                    values=[];view=copy.copy(base);view.finished=base.finished.clone()
                    for repeat in range(7):
                        view.episode_rows=[];torch.cuda.synchronize();tick=time.perf_counter()
                        function(view,ids,failure,applied);torch.cuda.synchronize()
                        if repeat>=2:values.append(time.perf_counter()-tick)
                    times[label]=statistics.median(values)
                episode_cases.append(dict(episodes=count,old_s=times['old'],new_s=times['new'],speedup=times['old']/times['new']))
            contact_times={}
            for label,kind in [('old',old_type),('new',new_type)]:
                values=[]
                for repeat in range(7):
                    obj=kind(physics,cfg);torch.cuda.synchronize();tick=time.perf_counter()
                    for _ in range(100):obj.update()
                    torch.cuda.synchronize()
                    if repeat>=2:values.append((time.perf_counter()-tick)/100)
                contact_times[label]=statistics.median(values)
            report.update(passed=True,episodes_exact=True,episode_timing=episode_cases,
                contact_timing=dict(old_s=contact_times['old'],new_s=contact_times['new'],speedup=contact_times['old']/contact_times['new']),
                scope='Identical synthetic GPU inputs only; no simulation, policy updates, full training speed or L40 speed claim.')
    except BaseException as exc:report['error']=repr(exc);raise
    finally:(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
