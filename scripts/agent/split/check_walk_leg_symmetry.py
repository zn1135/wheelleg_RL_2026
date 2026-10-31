#!/usr/bin/env python3
"""Walk symmetry geometry, unchanged reward terms/contracts and archived v6 poses; no learning."""
import argparse
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split import rewards, runner
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from scripts.agent.split.check_walk_command_readiness import archived
from scripts.agent.split.check_walk_tracking_economy import fake_inputs
from scripts.agent.split.check_chuanliantui_two_policy import digest, rejected


@torch.inference_mode()
def reward_checks(raw, before):
    s = raw.command_manager.get_term('standup'); n = raw.num_envs
    current, previous = rewards.WalkRewardState(raw), before.WalkRewardState(raw)
    assert len(rewards.WALK_WEIGHTS) == 18 and len(before.WALK_WEIGHTS) == 17
    assert {k:v for k,v in rewards.WALK_WEIGHTS.items() if k != 'nominal_state'} == before.WALK_WEIGHTS
    assert rewards.WALK_WEIGHTS['nominal_state'] == -3. and rewards.WALK_WEIGHTED_LIMITS['nominal_state'] == 10.
    assert rewards.GETUP_WEIGHTS == before.GETUP_WEIGHTS
    # getup配方与本任务起点的比较由contract_checks完成；旧17项快照早于并行getup修改。
    names = ('has_stood', 'has_landed', 'standing_time', 'reward_active', 'numerical_failure',
             'tracking_active', 'position_locked', 'observation_recovered')
    saved = {name: getattr(s, name).clone() for name in names}; rows = []
    try:
        s.reward_active.fill_(True); s.numerical_failure.zero_()
        current.step = -1
        # 真实复位姿态，不替换几何数据；尚未首次站稳时也会计算该项。
        nominal = current.compute(raw)['nominal_state'].clone()
        assert nominal.abs().max() < 1e-8, nominal
        for difference in (0., 10., 20., 33., 85., 105., 120.):
            fake = fake_inputs(raw, current, torch.full((n,), .2, device=raw.device),
                               raw.scene['robot'].data.projected_gravity_b.clone())
            data = current.robot.data
            # 使用非单位root旋转，验证世界系位置必须先转回机体系。
            quat = torch.zeros(n, 4, device=raw.device)
            quat[:, 0] = math.cos(math.radians(35.)/2.)
            quat[:, 2] = math.sin(math.radians(35.)/2.)
            data.root_link_quat_w = quat
            theta = torch.tensor([difference/2., -difference/2.], device=raw.device).deg2rad().expand(n, -1)
            relative = torch.stack((-theta.sin(), torch.zeros_like(theta), -theta.cos()), -1)*.25
            world = quat_apply(quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
                               relative.reshape(-1, 3)).reshape(n, 2, 3)
            data.body_link_pos_w[:, current.wheels] = data.body_link_pos_w[:, current.hips]+world
            previous.robot = current.robot
            for stood in (False, True):
                s.has_stood.fill_(stood); s.has_landed.fill_(stood)
                s.observation_recovered.fill_(stood); s.standing_time.fill_(.5 if stood else 0.)
                for active, locked in ((False, False), (True, False), (True, True)):
                    s.tracking_active.fill_(active); s.position_locked.fill_(locked)
                    current.step = previous.step = -1
                    new, old = current.compute(fake), previous.compute(fake)
                    assert all(torch.equal(new[name], old[name]) for name in before.WALK_WEIGHTS)
                    expected = max(-10., -3.*math.radians(difference)**2)
                    weighted = new['nominal_state']*-3.
                    assert torch.allclose(weighted, torch.full_like(weighted, expected), atol=3e-6), (difference, weighted)
            rows.append(dict(difference_deg=difference, weighted=float(weighted[0]), expected=expected))
            for flag, value in (('numerical_failure', True), ('reward_active', False)):
                getattr(s, flag).fill_(value); current.step = -1
                assert not current.compute(fake)['nominal_state'].any()
                getattr(s, flag).fill_(not value)
        return dict(cases=rows, nominal_reset_raw_max=float(nominal.max()),
                    nominal_reset_weighted_min=float((nominal*-3.).min()),
                    all_other_17_terms_bitwise_equal=True, active_before_first_stand=True,
                    numerical_and_inactive_masks=True, nonidentity_root_rotation=True,
                    getup_weights_and_recipe_unchanged=True, saturation_difference_deg=math.degrees(math.sqrt(10./3.)))
    finally:
        for name, value in saved.items():getattr(s, name).copy_(value)


@torch.inference_mode()
def trajectory_check(trajectory, scene):
    model = mujoco.MjModel.from_binary_path(str(scene)); data = mujoco.MjData(model)
    hips = [model.body(f'robot/{name}').id for name in ('lf0', 'rf0')]
    wheels = [model.body(f'robot/{name}').id for name in ('lfwheel', 'rfwheel')]
    with np.load(trajectory) as archive:
        pose, times, valid = archive['qpos'], archive['time_s'], archive['valid']
        names = archive['metric_names'].tolist(); metrics = archive['metrics']
        assert pose.shape[-1] == model.nq
        # 固定每0.2秒抽样，包含起步；各帧保留全部有效环境，不按结果筛选。
        stride = round(.2/float(times[1]-times[0]))
        frames = sorted(set(range(0, len(times), stride)) | {len(times)-1})
        rows = []
        for frame in frames:
            for env_id in np.flatnonzero(valid[frame]):
                data.qpos[:] = pose[frame, env_id]; mujoco.mj_forward(model, data)
                relative = torch.from_numpy((data.xpos[wheels]-data.xpos[hips]).copy())
                quat = torch.from_numpy(data.qpos[3:7].copy()).expand(2, -1)
                relative = quat_apply_inverse(quat, relative)
                swing = torch.atan2(-relative[:, 0], -relative[:, 2]).rad2deg().numpy()
                m = metrics[frame, env_id]
                rows.append(dict(frame=frame, env=int(env_id), time_s=float(times[frame]),
                    left_deg=float(swing[0]), right_deg=float(swing[1]), difference_deg=float(abs(swing[0]-swing[1])),
                    requested_vx=float(m[names.index('request_vx')]), requested_yaw=float(m[names.index('request_yaw')])))
    after = [row for row in rows if row['time_s'] >= 1.]
    def stats(selected):
        if not selected:return None
        difference = np.array([row['difference_deg'] for row in selected])
        return dict(samples=len(selected), mean_deg=float(difference.mean()), p95_deg=float(np.percentile(difference, 95)),
                    left_mean_deg=float(np.mean([row['left_deg'] for row in selected])),
                    right_mean_deg=float(np.mean([row['right_deg'] for row in selected])))
    return dict(trajectory=str(trajectory), trajectory_sha256=digest(trajectory), scene_sha256=digest(scene),
        selected_frames=frames, selection='Uniform 0.2s samples including startup, all valid envs; no held post-failure poses',
        all_valid=stats(rows), zero_motion_request=stats([r for r in rows if r['requested_vx'] == r['requested_yaw'] == 0.]),
        after_1s=stats(after), zero_vx_request_after_1s=stats([r for r in after if r['requested_vx'] == 0.]),
        zero_motion_request_after_1s=stats([r for r in after if r['requested_vx'] == r['requested_yaw'] == 0.]),
        initial=stats([row for row in rows if row['time_s'] == 0.]), rows=rows)


def contract_checks(args):
    baseline = json.loads(args.before_contracts.read_text())
    current = json.loads(json.dumps({role: runner.contracts(command_cfg(role)) for role in ('walk', 'getup')}))
    assert current['getup'] == baseline['getup']
    assert current['walk']['experiment_profile_contract'] == 'chuanliantui-walk-physical25-v8'
    assert current['walk']['reward_recipe'] == baseline['walk']['reward_recipe']
    assert current['walk']['reward_weights'] == baseline['walk']['reward_weights']
    old = torch.load(args.previous_v7, map_location='cpu', weights_only=False)['infos']
    rejected(lambda: runner.validate(old, command_cfg('walk')))
    valid = dict(runner.contracts(command_cfg('walk')))
    # 使用当前有效课程，分别证明缺项和旧recipe足以独立拒载。
    from ct_mjlab.envs.chuanliantui_split.mdp import walk_speed_config
    valid['standup_curriculum'] = dict(unlocked=True, speed_curriculum=dict(
        config=walk_speed_config(command_cfg('walk')),
        **{d:dict(stage=0,segments=0,successes=0,last_success_fraction=0.) for d in ('forward','backward')}),
        yaw_control={k:getattr(command_cfg('walk'),k) for k in ('yaw_rate_limit','yaw_acceleration','heading_gain')},
        position_hold={k:getattr(command_cfg('walk'),k) for k in ('position_gain','position_velocity_limit',
        'position_tolerance','position_lock_speed','position_lock_duration')})
    runner.validate(valid, command_cfg('walk'))
    no_term = copy.deepcopy(valid); no_term['reward_weights'].pop('nominal_state')
    rejected(lambda: runner.validate(no_term, command_cfg('walk')))
    old_recipe = copy.deepcopy(valid); old_recipe['reward_recipe'] = 'split-walk-relative-velocity-parking-noboost-tilt5-v4'
    rejected(lambda: runner.validate(old_recipe, command_cfg('walk')))
    return dict(profile_v8=True, getup_contract_unchanged=True, v7_recipe_and_weights_unchanged=True,
                old_v7_rejected=True, missing_symmetry_and_old_recipe_independently_rejected=True, current=current)


def main(args):
    torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=False)
    report = dict(verified=False, parameter_updates=0, scope='仅软件/合成检查，未训练、未行为验收')
    try:
        before = archived(args.before_rewards, 'ct_mjlab.envs.chuanliantui_split._symmetry_before')
        report['contracts'] = contract_checks(args)
        env, _ = task_registry.make_env('chuanliantui_walk', play=True, num_envs=16, seed=43, device=args.device)
        try:report['rewards'] = reward_checks(env.unwrapped, before)
        finally:env.close()
        report['trajectory'] = trajectory_check(args.trajectory, args.scene)
        report['verified'] = True
    finally:
        report['source_sha256'] = digest(__file__); report['before_sha256'] = digest(args.before_rewards)
        (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:(dict(v, rows='see result.json') if k == 'trajectory' else v)
                      for k,v in report.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before-rewards', 'before-contracts', 'previous-v7', 'trajectory', 'scene', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Symmetry check cannot train')):
        main(parser.parse_args())
