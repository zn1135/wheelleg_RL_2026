"""真实扫描与重复高度的同轮数GPU对照；保持训练进程及预算，禁止评估更新。"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

from . import inspect_regions_stage as regional
from . import inspect_randomization_4000 as frozen
from .run_torque_ablation import run_command

ROOT = frozen.ROOT
MODULE = 'scripts.agent.fudan_ppo_walk.inspect_critic_scan_stage'
PYTHON = ROOT/'mjlab_training/.venv/bin/python'
ARMS = dict(
    control=(ROOT/'logs/mjlab/fudan_ppo_walk/20261116_regions10to20_resume5000_to8000_01',
             ROOT/'artifacts/fudan_ppo_walk/20261116-regions-resume5000-8000-01'),
    scan=(ROOT/'logs/mjlab/fudan_ppo_walk/20261116_critic_scan_regions_resume5000_to6000_01',
          ROOT/'artifacts/fudan_ppo_walk/20261116-critic-scan-resume5000-6000-01'))


def configure(arm):
    regional.RUN, regional.EXPERIMENT = ARMS[arm]
    frozen.RUN = regional.RUN


def audit(out, iteration, arm):
    import torch
    configure(arm)
    regional.audit(out/arm, iteration)
    b = json.loads((regional.EXPERIMENT/'budget.json').read_text())
    saved = torch.load(out/arm/'checkpoint'/f'model_{iteration}.pt', map_location='cpu', weights_only=False)
    profile = saved.get('critic_observation', dict(version='fudan-critic-repeated-height-v1'))
    assert profile['version'] == ('fudan-critic-yaw-height-grid-v1' if arm == 'scan' else 'fudan-critic-repeated-height-v1')
    if arm == 'scan':
        assert profile == b['critic_observation']
    frozen.write(out/arm/'critic-identity.json', dict(passed=True, arm=arm, critic_observation=profile,
        initialization_sha256=frozen.sha(regional.RUN/'model_initial.pt'), optimizer_updates=0))


def worker(out, iteration, arm, stage, recovery=False):
    import torch
    configure(arm)
    proof = json.loads((out/arm/'checkpoint-audit.json').read_text())
    checkpoint = out/arm/'checkpoint'/f'model_{iteration}.pt'
    assert proof['passed'] and frozen.sha(checkpoint) == proof['checkpoint_sha256']
    # 每组单独进程，严格执行各自已校验的训练源码，不混用旧/新模块缓存。
    assert not any(k == 'ct_mjlab' or k.startswith('ct_mjlab.') for k in sys.modules)
    sys.meta_path.insert(0, frozen.FrozenTrainingSource())
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.fudan_ppo_walk import policy as loading
    original, load = physical.make_env_cfg, loading.load_policy
    networks = []
    destination = out/arm
    if recovery:
        assert stage == 'height-sweep', 'Only the interrupted scan height batch is recovered'
        destination = destination/'recovery-01'
        destination.mkdir(exist_ok=False)
    def physical_config(*args, **kwargs):
        cfg = original(*args, **kwargs)
        cfg.sim.nconmax, cfg.sim.njmax = 64, 256
        return cfg
    def load_checked(*args, **kwargs):
        policy, meta = load(*args, **kwargs)
        networks.append((policy, {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}))
        return policy, meta
    with patch.object(physical, 'make_env_cfg', physical_config), \
         patch.object(loading, 'load_policy', load_checked), \
         patch('torch.optim.Adam.step', side_effect=AssertionError('Frozen evaluation cannot learn')):
        if stage in ('terrain', 'height-sweep'):
            regional.terrain(destination, iteration, height_sweep=stage == 'height-sweep',
                             models=[(str(iteration), checkpoint)])
        else:
            getattr(regional, stage)(out/arm, iteration)
    assert networks
    assert all(torch.equal(v.cpu(), before[k]) for policy, before in networks for k,v in policy.state_dict().items())
    assert frozen.sha(checkpoint) == proof['checkpoint_sha256']
    frozen.write(destination/(stage+'-frozen-execution.json'), dict(passed=True, modules=frozen.LOADED,
        actual_networks_unchanged=True, loaded_networks=len(networks), optimizer_updates=0,
        checkpoint_unchanged=True, training_not_interrupted_by_evaluation=True,
        training_host_interruption_recorded=(ARMS['scan'][1]/'host-interruption.json').exists()))


def terrain_path(out, arm, folder, iteration):
    recovered = out/arm/'recovery-01'/folder/str(iteration)
    return recovered if (recovered/'result.json').exists() else out/arm/folder/str(iteration)


def summarize(out, iteration):
    import numpy as np
    from scripts.agent.motion.acceptance import compare_baseline
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    result = dict(iteration=iteration, optimizer_updates=0, training_budget_unchanged=True,
        behavior_accepted=False, nominal={}, terrain={}, extras={}, training={})
    for arm in ARMS:
        root = out/arm
        nominal = json.loads((root/'nominal/acceptance/acceptance.json').read_text())
        result['nominal'][arm] = dict(passed=nominal['technical_passed'], total=nominal['total'])
        for label in ('high-yaw', 'speed3'):
            score = json.loads((root/label/'acceptance.json').read_text())
            result['extras'].setdefault(label, {})[arm] = dict(passed=score['technical_passed'], total=score['total'])
        rows = []
        for folder in ('terrain', 'terrain-height-extra'):
            terrain = terrain_path(out, arm, folder, iteration)
            data = json.loads((terrain/'result.json').read_text())
            rows += data['rows']
            for row in data['rows']:
                with np.load(terrain/'traces'/(row['id']+'.npz')) as trace:
                    assert all(np.isfinite(trace[k]).all() for k in trace.files)
                    assert abs(trace['time'][-1]-trace['qpos_time'][-1]) < 1e-8
                    if row['complete']:
                        assert abs(trace['time'][-1]-row['duration']) < 1e-8
                    else:
                        assert trace['terminated'][-1]
        result['terrain'][arm] = {family:dict(total=sum(r['family']==family for r in rows),
            reached=sum(r['reached'] for r in rows if r['family']==family),
            passed=sum(r['passed'] for r in rows if r['family']==family),
            fell=sum(r['fell'] for r in rows if r['family']==family)) for family in ('up','down','rough')}
        events = EventAccumulator(str(ARMS[arm][0]), size_guidance={'scalars':0}).Reload()
        reward = [v.value for v in events.Scalars('Train/reward') if v.step <= iteration]
        speed = [v for v in events.Scalars('Train/speed_limit') if v.step <= iteration]
        result['training'][arm] = dict(reward_first100=float(np.mean(reward[:100])),
            reward_last100=float(np.mean(reward[-100:])), speed_limit=speed[-1].value)
    left = json.loads((out/'control/nominal/acceptance/acceptance.json').read_text())
    right = json.loads((out/'scan/nominal/acceptance/acceptance.json').read_text())
    compare_baseline(right, left)
    result['nominal_scan_vs_control'] = right['comparison']
    compared, differences = 0, []
    for folder in ('terrain', 'terrain-height-extra'):
        a,b = [terrain_path(out, arm, folder, iteration) for arm in ('control','scan')]
        assert json.loads((a.parent.parent/(folder+'-protocol.json')).read_text()) == json.loads((b.parent.parent/(folder+'-protocol.json')).read_text())
        for path in sorted((a/'fixtures').glob('*_initial.npz'))+[a/'initial.npz']:
            with np.load(path) as x, np.load(b/path.relative_to(a)) as y:
                assert set(x.files)==set(y.files)
                for key in x.files:
                    compared += 1
                    if not np.array_equal(x[key], y[key]):
                        differences.append(dict(path=str(path.relative_to(a)), field=key,
                            max_difference=float(np.abs(x[key]-y[key]).max())))
    result['terrain_pairing'] = dict(initial_arrays_compared=compared, bitwise_equal=not differences, differences=differences)
    frozen.write(out/'comparison.json', result)
    lines = [f'# Critic扫描{iteration}阶段同轮数对照', '',
        f'冻结两组{iteration}模型，新增训练预算保持至6000；这是阶段效果，尚非最终6000成绩。', '',
        '| 指标 | 无扫描 | 77点扫描 |', '|---|---|---|']
    for label, field in [('原47平地', result['nominal']), ('高速yaw', result['extras']['high-yaw']), ('±3m/s',result['extras']['speed3'])]:
        lines.append('| '+label+' | '+' | '.join(f'{field[a]["passed"]}/{field[a]["total"]}' for a in ARMS)+' |')
    for family, label in [('up','上坡到目标'),('down','下坡到目标'),('rough','崎岖到目标')]:
        lines.append('| '+label+' | '+' | '.join(f'{result["terrain"][a][family]["reached"]}/{result["terrain"][a][family]["total"]}' for a in ARMS)+' |')
    lines += ['', '地形到目标不等于完整通过：另要求速度/姿态跟踪、停车及双轮支撑。所有采集为名义物理、无外载/推扰，0学习。', '',
              '四坡10～20°，三高度0.22/0.26/0.30m、速度0.5/1m/s，上下坡各24例；崎岖0.26m高16例。4m给定方向通行后停车3s，原共同运动门槛不放宽。', '',
              f'两组地形初始数组比较{compared}项，逐值相同：{not differences}。平地使用原统一验收及初态cohort配对；未测全随机987、起立或真机。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('audit','nominal','extras','terrain','height-sweep','run','report'))
    p.add_argument('--arm', choices=tuple(ARMS))
    p.add_argument('--recovery', action='store_true', help='Keep interrupted height batch; collect in recovery-01')
    p.add_argument('--iteration', type=int, default=5700)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();out = a.output.resolve()
    if a.mode == 'audit':
        return audit(out, a.iteration, a.arm)
    if a.mode == 'report':
        return summarize(out, a.iteration)
    if a.mode != 'run':
        return worker(out, a.iteration, a.arm, a.mode, a.recovery)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'inspection-source.py')
    write = frozen.write
    write(out/'plan.json', dict(kind='same_iteration_scan_ablation', iteration=a.iteration,
        stages=['audit','nominal','extras','terrain','height-sweep'], arms={k:[str(x) for x in v] for k,v in ARMS.items()},
        optimizer_updates=0, original_training_not_interrupted=True, created_unix=time.time()))
    try:
        for arm in ARMS:
            for stage in ('audit','nominal','extras','terrain','height-sweep'):
                run_command(out, arm+'-'+stage, [str(PYTHON),'-u','-m',MODULE,stage,'--arm',arm,
                    '--iteration',str(a.iteration),'--output',str(out)])
        summarize(out, a.iteration)
        write(out/'status.json', dict(status='evaluation_complete', optimizer_updates=0))
    except BaseException as exc:
        write(out/'status.json', dict(status='failed', error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
