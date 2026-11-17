"""Exact old/new contact and episode-record checks on identical GPU inputs; no learning."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch


def legacy_rows(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StepEnv')
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    block = next(n for n in step.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'len(new_ids)')
    function = ast.parse('def record(self, new_ids, failure, contact_failure):\n    pass').body[0]
    function.body = block.body
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['record']


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--steps', type=int, default=250)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg
    from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
    original = args.baseline_source/'mjlab_training/src/ct_mjlab/envs/fudan_ppo_step'
    spec = importlib.util.spec_from_file_location('old_contacts', original/'contacts.py')
    old_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_module)
    record_old = legacy_rows(original/'env.py')
    report = dict(passed=False, optimizer_updates=0, exact_contact_checks=0, episode_checks=0)
    env = None; torch.set_num_threads(1); torch.manual_seed(1)
    fields = ('ticks','count','streak','longest','peak','peak_pos','any_streak','strong_streak','failed',
              'step_contact','supported','side_contact','step_speed_peak','capacity_overflow','constraint_peak','global_contact_peak')
    try:
        with patch('torch.optim.Adam.step', side_effect=AssertionError('Optimization check cannot train')):
            env = StepEnv(20, seed=1, cfg=FudanStepCfg(), randomized=True)
            old = old_module.ContactMonitor(env.env, env.cfg.step)
            update = env.contacts.update; reset = env.contacts.reset
            def both_reset(ids):
                old.reset(ids); reset(ids)
            def checked_update():
                if time.time() >= args.deadline: raise TimeoutError('Equivalence deadline')
                old.record_substeps = env.contacts.record_substeps
                a = old.update(); b = update()
                equal = [(getattr(old,key)==getattr(env.contacts,key)).all() for key in fields]
                equal.append((a==b).all())
                assert bool(torch.stack(equal).all()), [key for key in fields if not torch.equal(getattr(old,key),getattr(env.contacts,key))]
                assert old.total_ticks == env.contacts.total_ticks
                report['exact_contact_checks'] += 1
                return b
            env.contacts.reset = both_reset; env.contacts.update = checked_update
            original_step = env.step
            def paired_step(*a, **kw):
                # 旧监视器也须经过同一策略拍的清零，否则比较的是不同时间窗口。
                old.step_contact.zero_(); old.step_speed_peak.zero_()
                old.step_forces.clear(); old.step_positions.clear()
                return original_step(*a, **kw)
            env.step = paired_step
            with torch.inference_mode():
                env.reset()
                for step in range(args.steps):
                    env.contacts.record_substeps = step % 2 == 0
                    env.step(torch.randn(20,6,device=env.device).clamp(-1,1))
                    if env.contacts.record_substeps:
                        assert torch.equal(old.step_forces[-1],env.contacts.step_forces[-1])
                        assert torch.equal(old.step_positions[-1],env.contacts.step_positions[-1])
                    old.step_forces.clear(); old.step_positions.clear()
                # 相同真实状态额外注入缺测标记，检查罕见诊断字段也逐值保留。
                ids = torch.arange(20,device=env.device)
                for missing in (False, True):
                    left, right = copy.copy(env), copy.copy(env)
                    for view in (left,right):
                        view.finished = env.finished.clone(); view.episode_rows = []
                        view.ray_failed = torch.full_like(env.ray_failed,missing)
                    failure=torch.rand(20,device=env.device)>.5
                    applied=torch.rand(20,device=env.device)>.5
                    record_old(left,ids,failure,applied)
                    right._record_episodes(ids,failure,applied)
                    assert left.episode_rows == right.episode_rows
                    assert torch.equal(left.finished,right.finished)
                    report['episode_checks'] += len(ids)
                # 只计格式化开销；两种方法使用同一GPU状态，数据不会反馈控制。
                samples=[]
                for count in (1,4,20):
                    chosen=ids[:count]; timings={}
                    for name,function in [('old',record_old),('new',StepEnv._record_episodes)]:
                        values=[]
                        view=copy.copy(env); view.finished=env.finished.clone();view.ray_failed=torch.zeros_like(env.ray_failed)
                        for repeat in range(12):
                            view.episode_rows=[];torch.cuda.synchronize();tick=time.perf_counter()
                            function(view,chosen,failure,applied);torch.cuda.synchronize()
                            if repeat>=2:values.append(time.perf_counter()-tick)
                        timings[name]=statistics.median(values)
                    samples.append(dict(episodes=count,old_s=timings['old'],new_s=timings['new'],speedup=timings['old']/timings['new']))
            report.update(passed=True, physics_ticks=env.contacts.total_ticks, all_contact_fields_bitwise_equal=True,
                all_episode_fields_exact=True, recording_substeps_checked=True, missing_ray_diagnostics_checked=True,
                episode_timing=samples, constraint_peak=int(env.contacts.constraint_peak),
                scope='20-world GPU same-input bitwise contact and exact episode rows; not stair behavioral acceptance.')
    except BaseException as exc: report['error']=repr(exc);raise
    finally:
        if env is not None:env.close()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
