"""冻结5200，对照0/2 m/s下0.22和0.26m机高命令，GPU采集后渲染。"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from .evaluate import policies, WALK
from .media import render
from .probe_takeoff_work import CHECKPOINT, CHECKPOINT_SHA, collect_profile, write_json


def run(output):
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    output.mkdir(parents=True, exist_ok=False)
    walk, jump, _ = policies(CHECKPOINT, 'cuda:0')
    networks = [(p, {k: v.cpu().clone() for k, v in p.state_dict().items()}) for p in (walk, jump)]
    result = dict(checkpoint_sha256=CHECKPOINT_SHA, walk_sha256=sha(WALK),
                  backend='mujoco_warp_gpu', worlds=20, seed=823, duration_s=6.,
                  switch_s=2., handoff_ms=80, optimizer_updates=0, groups={}, comparisons={})
    with patch('torch.optim.Adam.step', side_effect=RuntimeError('Frozen evaluation')):
        for speed in (2., 0.):
            for height in (.26, .22):
                key = f'v{speed:g}_h{height:.2f}'
                folder = output/key
                folder.mkdir()
                print('Collecting', key, flush=True)
                summary = collect_profile(folder/'probe', walk, jump, probe=True, height=height, speed=speed)
                cases = json.loads((folder/'probe/cases.json').read_text())
                for case in cases:
                    with np.load(folder/'probe'/f'start{case["id"]:02d}.npz') as raw:
                        assert np.allclose(raw['history0'].reshape(5, 25)[:, 8], height*5)
                        assert np.allclose(raw['history0'].reshape(5, 25)[:, 6], speed*3)
                        jumps = case['jumps']
                        case['landing_nonwheel_contact_ms'] = None
                        if jumps:
                            land = round(1000*(jumps[0]['takeoff_s']+jumps[0]['airtime_s']))
                            if land+1000 <= len(raw['trace']):
                                case['landing_nonwheel_contact_ms'] = int((raw['trace'][land:land+1000, 10]>1.).sum())
                first = [c['jumps'][0] for c in cases if c['jumps']]
                summary.update(command=[speed, 0., height], cases=cases,
                               first_peak_median_m=float(np.median([j['peak_lower_wheel_m'] for j in first])) if first else None,
                               first_peak_max_m=max((j['peak_lower_wheel_m'] for j in first), default=None),
                               landing_nonwheel_contact_cases=sum(c['landing_nonwheel_contact_ms'] not in (None, 0) for c in cases),
                               missing_landing_window_cases=sum(c['landing_nonwheel_contact_ms'] is None for c in cases))
                result['groups'][key] = summary
                write_json(output/'result.json', result)
    assert all(torch.equal(v.cpu(), before[k]) for p, before in networks for k, v in p.state_dict().items())
    for speed in (2., 0.):
        baseline = result['groups'][f'v{speed:g}_h0.26']
        candidate = result['groups'][f'v{speed:g}_h0.22']
        pairs = []
        for i in range(20):
            a = output/f'v{speed:g}_h0.26/probe/start{i:02d}.npz'
            b = output/f'v{speed:g}_h0.22/probe/start{i:02d}.npz'
            with np.load(a) as old, np.load(b) as new:
                assert np.array_equal(old['qpos0'], new['qpos0']) and np.array_equal(old['qvel0'], new['qvel0'])
                history = old['history0'].reshape(5, 25).copy()
                history[:, 8] = new['history0'].reshape(5, 25)[:, 8]
                assert np.array_equal(history.flatten(), new['history0'])
            old_jumps, new_jumps = baseline['cases'][i]['jumps'], candidate['cases'][i]['jumps']
            difference = new_jumps[0]['peak_lower_wheel_m']-old_jumps[0]['peak_lower_wheel_m'] if old_jumps and new_jumps else None
            pairs.append(dict(id=i, initial_physical_state_exact=True, peak_difference_m=difference))
        result['comparisons'][str(speed)] = pairs
        eligible = [c for c in candidate['cases'] if c['jumps']]
        ordered = sorted(eligible, key=lambda c: c['jumps'][0]['peak_lower_wheel_m'])
        selected = ordered[(len(ordered)-1)//2]['id'] if ordered else 0
        folder = output/f'v{speed:g}_h0.22'
        with np.load(folder/'probe'/f'start{selected:02d}.npz') as raw:
            poses = raw['qpos_pre'][::40, None, :]
        model = mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
        render(model, poses, folder/'replay.mp4')
        candidate['video_case'] = selected
        candidate['video_frames'] = len(poses)
    result['network_parameters_unchanged'] = True
    result['source_sha256'] = sha(Path(__file__))
    write_json(output/'result.json', result)
    lines = ['# Jump5200机高命令对照', '',
             '0.22和0.26m分别用于WALK助跑/站立及Jump全过程，五帧历史从起始即使用对应高度。'
             '每组20个相同物理初态、6秒GPU物理，2秒时切入Jump，触轮后80ms交回WALK，0学习。', '',
             '| 前进指令 | 机高命令 | 起跳 | 首跳轮底峰值中位 | 最高 | 落地首秒非轮接触 | 连续双轮支撑1秒 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for group in result['groups'].values():
        speed, _, height = group['command']
        median = f'{group["first_peak_median_m"]*100:.2f} cm' if group['first_peak_median_m'] is not None else '缺测'
        highest = f'{group["first_peak_max_m"]*100:.2f} cm' if group['first_peak_max_m'] is not None else '缺测'
        lines.append(f'| {speed:g} m/s | {height:.2f} m | {group["cases_with_takeoff"]}/20 | {median} | {highest} | {group["landing_nonwheel_contact_cases"]}/20 | {group["first_landing_stable_cases"]}/20 |')
    lines += ['', '峰值按同一时刻较低轮底统计。0m/s为零速度指令，机器人实际速度由闭环产生。'
              '本次采用仿真接触切模，实机的支持力估计切模和实际传感状态需另行对齐。'
              '已有同初态GPU普通重复出现约2cm逐例峰值波动，本次每条件一批，厘米级差异应结合该波动判断。', '',
              '[2m/s、0.22m中位高度样本视频](v2_h0.22/replay.mp4) · [0m/s、0.22m中位高度样本视频](v0_h0.22/replay.mp4) · [完整逐例数据](result.json)', '']
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k: {n: g[n] for n in ('cases_with_takeoff', 'first_peak_median_m', 'first_peak_max_m', 'first_landing_stable_cases')} for k, g in result['groups'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
