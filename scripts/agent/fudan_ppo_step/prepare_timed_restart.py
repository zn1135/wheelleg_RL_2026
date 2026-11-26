"""从固定速度 5200 派生 v12，并检查完整计时时序与平地巡航。"""
import argparse
import copy
import math
import time
from pathlib import Path
from unittest.mock import PropertyMock, patch

import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import FudanStepCfg, TIMED_RECIPE, recipe_record
from .launch_training import sha, write
from .prepare_restart import SOURCE, SOURCE_SHA
from .remote_training import same_state


def prepare(output):
    assert sha(SOURCE) == SOURCE_SHA
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert saved['task'] == 'fudan_ppo_step' and not saved['partial_update']
    assert (saved['iterations'], saved['transitions'], saved['num_envs']) == (5200, 1022361600, 4096)
    assert saved['primary_updates'] == saved['encoder_updates'] == 104000
    before = saved['training_recipe']
    assert before['step']['version'] == 'fixed200-straight-2mps-v10'
    expected = copy.deepcopy(before)
    expected['step'].update(version=TIMED_RECIPE, restart_flat_probability=.5, delayed_stair_probability=.25,
        restart_initial_wait_s=[.75, 2.], restart_first_drive_s=2.5, restart_pause_s=[1.5, 3.],
        restart_second_drive_s=3., flat_spawn_yaw_deg=0., flat_boundary_timeout=True, down_lane_until_edge=True)
    assert expected == recipe_record(FudanStepCfg(TIMED_RECIPE))
    migration = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, before_recipe=before, after_recipe=expected,
        previous_migration=saved.get('recipe_migration'), learning_state_reset=False,
        terrain_assignment_preserved=True, physical_episodes_reinitialized=True)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'resume5200_timed_restart.pt'
    torch.save(dict(saved, training_recipe=expected, recipe_migration=migration), path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    assert same_state(restored['training_recipe'], expected) and same_state(restored['recipe_migration'], migration)
    for key in saved.keys() - {'training_recipe', 'recipe_migration'}:
        assert same_state(saved[key], restored[key]), key
    write(output/'migration.json', dict(migration, passed=True, derived_sha256=sha(path), optimizer_updates=0))


def check_timing(env):
    ids = torch.arange(env.num_envs, device=env.device)
    # 用真实抽样的等待回合，在速度高于门槛时检查每个时段前一拍与到点拍。
    for _ in range(64):
        env.resample(ids)
        waiting = env.command_mode == env.WAIT_TO_START
        flat = waiting & (env.modes == 0)
        up = waiting & (env.modes == 1)
        if flat.any() and up.any():
            break
    assert flat.any() and up.any() and not waiting[env.modes == 2].any()
    assert ((env.stop_command_step[waiting]>=75)&(env.stop_command_step[waiting]<=200)).all()
    assert ((env.restart_pause_ticks[flat]>=150)&(env.restart_pause_ticks[flat]<=300)).all()
    data = env.robot.data
    moving = torch.ones_like(data.root_link_lin_vel_w)
    with patch.object(type(data), 'root_link_lin_vel_w', new_callable=PropertyMock, return_value=moving) as velocity:
        env.command_elapsed[waiting] = env.stop_command_step[waiting]-1
        env._advance_restart_commands()
        assert env.commands[waiting, 0].eq(0).all()
        env.command_elapsed[waiting] += 1
        env._advance_restart_commands()
        assert env.commands[waiting, 0].eq(2).all() and env.command_mode[up].eq(0).all()
        for mode, duration, next_mode, target in (
            (env.FIRST_DRIVE, 250, env.WAIT_TO_RESTART, 0.),
            (env.WAIT_TO_RESTART, env.restart_pause_ticks[flat], env.SECOND_DRIVE, 2.),
            (env.SECOND_DRIVE, 300, env.FINAL_STOP, 0.)):
            env.command_elapsed[flat] = duration-1
            env._advance_restart_commands()
            assert env.command_mode[flat].eq(mode).all()
            env.command_elapsed[flat] += 1
            env._advance_restart_commands()
            assert env.command_mode[flat].eq(next_mode).all() and env.commands[flat, 0].eq(target).all()
        env.command_elapsed[flat] = env.max_episode_length
        env._advance_restart_commands()
        assert env.commands[flat, 0].eq(0).all() and env.command_mode[flat].eq(env.FINAL_STOP).all()
        velocity.assert_not_called()


def check_boundaries(env):
    # 在真实 step/终止/回合记录路径注入确定的几何，采样完终止拍再复位。
    env.reset()
    flat, up, down = [(env.modes == m).nonzero().flatten() for m in (0, 1, 2)]
    with patch.object(env.env, 'step'), patch.object(env, '_update_state'):
        origin = env.env.scene.env_origins
        wheels = env.env.sim.data.xpos
        pos = wheels[:, env.wheel_ids].clone()
        pos[down, :, 1] = origin[down, None, 1]+.60
        edge = origin[down, 0] - env.cfg.step.approach_distance[2]
        pos[down, :, 0] = edge[:, None]-.35
        # 同一批分别覆盖未完全越沿、双轮越沿后、上台越界、上台过远。
        pos[down[0], 0, 0] = edge[0]+.02
        pos[up[0], :, 1] = origin[up[0], 1]+.60
        pos[up[:2], :, 0] = origin[up[:2], None, 0]+env.cfg.step.approach_distance[1]+.3
        pos[up[1], :, 0] += env.cfg.step.depth
        wheels[:, env.wheel_ids] = pos
        env.position = env.position.clone()
        env.position[flat, 1] = origin[flat, 1]+1.61
        env.contacts.failed[flat[1]] = True
        env.position[down[2], 1] = origin[down[2], 1]+1.61
        env.contacts.supported[down] = True
        env.support_height[down] = 0.
        env.landing_ticks[down] = math.ceil(env.cfg.step.landing_s/env.dt)-1
        _, _, _, done, info, _ = env.step(torch.zeros_like(env.action.raw_action), auto_reset=False, resample=False)
        rows = {r['env']:r for r in env.episode_rows}
        for i in (flat[0], flat[1], up[0], up[1], down[0], down[2]):
            assert done[i]
        assert info['time_outs'][flat[0]] and not info['step_failure'][flat[0]]
        assert rows[int(flat[0])]['flat_boundary_timeout'] and not rows[int(flat[0])]['failure']
        assert info['step_failure'][flat[1]] and not info['time_outs'][flat[1]]
        assert not rows[int(flat[1])]['flat_boundary_timeout']
        assert info['step_failure'][up[:2]].all() and not info['time_outs'][up[:2]].any()
        assert rows[int(down[0])]['down_lane_failure'] and not info['time_outs'][down[0]]
        assert env.crossed[down[1]] and not done[down[1]]
        assert info['step_failure'][down[2]] and not rows[int(down[2])]['down_lane_failure']
        pos[down[1], 0, 0] = edge[1]+.02
        wheels[:, env.wheel_ids] = pos
        env._progress()
        assert not env.down_lane_failure[down[1]]
    env.reset()
    assert not env.down_edge_passed.any()
    env.episode_rows.clear()


@torch.no_grad()
def check_timed(env, args):
    import numpy as np
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    check_timing(env)
    check_boundaries(env)
    flat = env.modes == 0
    # 实际复位验证评估覆盖仍优先于训练朝向。
    env.env._step_fixed_yaw = torch.full_like(env.position[:, 0], .17)
    env.reset_idx(flat.nonzero().flatten())
    quat = env.env.sim.data.qpos[flat, 3:7]
    assert torch.allclose(2*torch.atan2(quat[:, 3], quat[:, 0]), torch.full_like(quat[:, 0], .17), atol=1e-6)
    del env.env._step_fixed_yaw
    policy, _ = load_policy(SOURCE, device=env.device)
    cruise_ticks = round(env.cfg.step.flat_cruise_s/env.dt)
    for _ in range(64):
        obs, _, history = env.reset()
        cruise = flat & (env.command_mode == 2) & (env.stop_command_step == cruise_ticks)
        if cruise.any():
            break
    assert cruise.any()
    quat = env.env.sim.data.qpos[flat, 3:7]
    yaw = torch.rad2deg(torch.atan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),
        1-2*(quat[:,2].square()+quat[:,3].square())))
    assert yaw.abs().max() < .1
    viewer = None
    trajectory = []
    try:
        if args.viewer:
            import viser
            from mjlab.viewer.viser.scene import MjlabViserScene
            viewer = viser.ViserServer(host='127.0.0.1', port=args.port, label='Step v12 · 20 GPU environments')
            scene = MjlabViserScene(viewer, env.env.sim.mj_model, env.num_envs,
                sim_model=env.env.sim.model, expanded_fields=env.env.sim.expanded_fields)
            scene.show_only_selected = False
            scene.create_scene_gui(camera_distance=12., show_debug_viz_control=False)
            print(f'20-environment GPU preview: http://127.0.0.1:{viewer.get_port()}', flush=True)
        while env.command_elapsed[cruise].min().item() < cruise_ticks:
            if time.time() >= args.deadline:
                raise TimeoutError('Timed cruise preview exceeded deadline')
            assert env.commands[cruise, 0].eq(2.).all()
            trajectory.append((env.position[cruise,:2]-env.env.scene.env_origins[cruise,:2]).cpu().numpy())
            if viewer and len(trajectory)%5 == 1:
                scene.update(env.env.sim.data)
            action, _ = policy.act_inference(obs, history)
            obs, _, _, done, _, history = env.step(action)
            assert not done[cruise].any(), 'Flat cruise ended before its 7 s command completed'
        assert env.commands[cruise, 0].eq(0.).all()
        trajectory.append((env.position[cruise,:2]-env.env.scene.env_origins[cruise,:2]).cpu().numpy())
    finally:
        if viewer:
            viewer.stop()
    xy = np.stack(trajectory)
    np.savez_compressed(args.output/'flat-cruise.npz', xy_m=xy, env_ids=cruise.nonzero().flatten().tolist(),
                        flat_spawn_yaw_deg=yaw.tolist(), policy_dt=env.dt)
    env.reset()
    env.stop_totals.zero_()
    env.episode_rows.clear()
    return dict(moving_released_on_time=True, timed_pause_and_both_drives=True, final_zero_hold=True,
        boundary_bootstrap_and_failure_precedence=True, down_lane_before_and_after_edge=True,
        up_lane_and_too_far_retained=True, fixed_yaw_override=True, flat_reset_yaw_deg=yaw.tolist(),
        cruise_envs=int(cruise.sum()), cruise_s=env.cfg.step.flat_cruise_s, cruise_end_xy_m=xy[-1].tolist(),
        cruise_max_abs_y_m=float(abs(xy[:,:,1]).max()), cruise_lateral_truncations=0, viewer=bool(args.viewer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
