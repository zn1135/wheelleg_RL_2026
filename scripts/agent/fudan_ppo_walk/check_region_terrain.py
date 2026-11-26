"""20-world regional terrain geometry/GPU precheck; no network or learning."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, section_dict, timing_record
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.terrain import height_map, record, critic_record, RegionalTerrainCfg


def integration_checks(cfg, recipe, output):
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.terrain import PLANE
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfgPPO
    from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner
    from ct_mjlab.scripts.train import main as train_main
    from ct_mjlab.envs import task_registry
    assert record(FudanWalkCfg()) == PLANE
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import FudanStepCfg
    assert record(FudanStepCfg()) == PLANE
    stale = output/'plane-precheck-rejection-input.json'
    stale.write_text(json.dumps(dict(terrain=PLANE)))
    with patch.object(task_registry, 'make_env', side_effect=AssertionError('Must reject before GPU construction')):
        try:
            train_main(['--task', 'fudan_ppo_walk', '--terrain', 'regions', '--num-envs', '20',
                        '--iterations', '1', '--log-dir', str(output/'must-not-train'), '--precheck', str(stale)])
        except ValueError as exc:
            assert 'precheck terrain differs' in str(exc)
        else:
            raise AssertionError('Plane precheck accepted for regional training')
    assert not (output/'must-not-train').exists()
    rejected = None
    if recipe is not None and (recipe.parent/'model_final.pt').exists():
        source = recipe.parent/'model_final.pt'
        dummy = SimpleNamespace(env=SimpleNamespace(cfg=cfg), train_cfg=FudanWalkCfgPPO(), device='cpu')
        try:
            FudanRunner.load(dummy, source)
        except ValueError as exc:
            assert 'Terrain differs' in str(exc)
            rejected = str(source)
        else:
            raise AssertionError('Terrain migration accepted without explicit selection')
    return dict(plane_precheck_rejected_before_gpu=True, old_checkpoint_requires_explicit_terrain=rejected,
                old_walk_and_step_remain_plane=True, training_started=False)


def render_preview(output, terrain=None):
    from PIL import Image, ImageDraw, ImageFont
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    terrain = FudanWalkCfg().terrain if terrain is None else terrain
    model = mujoco.MjModel.from_binary_path(str(output/'fixtures/model.mjb'))
    disable_model_sameframe_shortcuts(model)
    # 大场地纯色材质在原强光下过曝；只调整离线预览副本的灯光和颜色。
    model.vis.headlight.ambient[:] = .15
    model.vis.headlight.diffuse[:] = 0.
    model.light_diffuse[:] = 0.
    model.light_diffuse[0] = .9
    model.light_dir[0] = [-1., 1., -.2]
    model.light_specular[:] = 0.
    for name, color in (('walk_flat', (.25, .4, .55, 1.)),
                        ('walk_pyramid_base', (.4, .5, .3, 1.)),
                        ('walk_rough', (.65, .42, .24, 1.)),
                        ('walk_outer_floor', (.12, .15, .18, 1.))):
        model.geom_rgba[model.geom(name).id] = color
    for i in range(4):
        model.geom_rgba[model.geom(f'walk_pyramid_{i}').id] = [.55, .7, .3, 1.]
    model.vis.global_.offwidth, model.vis.global_.offheight = 1100, 800
    data = mujoco.MjData(model)
    with np.load(output/'fixtures/initial.npz') as saved:
        data.qpos[:] = saved['qpos'][0]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [7., 7., .2]
    camera.distance, camera.elevation, camera.azimuth = 25., -28., 135.
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 0, 0, 0, 0, 0]
    with mujoco.Renderer(model, width=1100, height=800) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = Image.fromarray(renderer.render().copy())
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 25)
    draw.rectangle((0, 0, 1100, 84), fill='#18212b')
    draw.text((16, 8), '连续分区地形：平地50% / 金字塔坡25% / 小崎岖25%', font=font, fill='white')
    low, high = terrain.slope_degrees
    draw.text((16, 45), f'金字塔四面坡{low:g}～{high:g}°、顶部{terrain.pyramid_platform:g}m平顶；崎岖起伏±{terrain.rough_amplitude*100:g}cm', font=font, fill='white')
    image.save(output/'terrain-3d.png')


def geometric_checks(cfg, output):
    axis, heights, spawns = height_map(cfg.terrain)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name='terrain')
    RegionalTerrainCfg(terrain=cfg.terrain).function(0., spec, np.random.default_rng(1))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    half = cfg.terrain.size/2
    def height(x, y):
        return 2.-mujoco.mj_ray(model, data, np.array([x+half, y+half, 2.]),
            np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, None)
    spawn_error = max(abs(height(x, y)-z) for x, y, z in spawns)
    assert spawn_error < 1e-6
    slopes = []
    for x, y, z in spawns[4:8]:
        # 四个坡面都测真实碰撞表面，角点斜线不是坡面角度。
        angles = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rise = height(x+dx, y+dy)-height(x+2*dx, y+2*dy)
            angles.append(float(np.degrees(np.arctan(rise))))
        slopes.append(angles)
    np.testing.assert_allclose(slopes, np.repeat(np.linspace(*cfg.terrain.slope_degrees, 4)[:, None], 4, 1), atol=1e-5)
    assert heights[:, :len(axis)//2].max() == heights[:, :len(axis)//2].min() == 0.
    rough = heights[:len(axis)//2, len(axis)//2:]
    assert rough.min() >= -cfg.terrain.rough_amplitude-1e-12 and rough.max() <= cfg.terrain.rough_amplitude+1e-12
    # 三个分区接缝及四座金字塔坡脚应为0，不能附带人工台阶。
    mid = len(axis)//2
    assert np.abs(heights[:, mid]).max() < 1e-12
    assert np.abs(heights[mid, :]).max() < 1e-12
    assert np.abs(heights[[0, -1], :]).max() < 1e-12
    assert np.abs(heights[:, [0, -1]]).max() < 1e-12
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(heights, origin='lower', extent=(-half, half, -half, half), cmap='terrain')
    ax.plot([0, 0], [-half, half], 'k--', lw=1)
    ax.plot([0, half], [0, 0], 'k--', lw=1)
    ax.scatter(spawns[:, 0], spawns[:, 1], c='red', marker='x', label='Reset centres')
    ax.text(-8, 0, 'FLAT 50%', ha='center', fontsize=16)
    ax.text(8, 15, 'PYRAMID 25% / four-sided / 1m flat top', ha='center', fontsize=10)
    ax.text(8, -15, 'ROUGH 25% / +/-1cm / 0.2m spacing', ha='center', fontsize=10)
    for (x, y, _), slope in zip(spawns[4:8], np.linspace(*cfg.terrain.slope_degrees, 4)):
        ax.text(x, y+1, f'{slope:.2f} deg', ha='center')
    ax.set(xlabel='World X (m)', ylabel='World Y (m)', title='Continuous regional terrain (32 x 32 m)')
    ax.legend(loc='upper left')
    fig.colorbar(image, ax=ax, label='Ground elevation (m)')
    fig.tight_layout()
    fig.savefig(output/'terrain-layout.png', dpi=150)
    plt.close(fig)
    np.savez_compressed(output/'terrain-heightmap.npz', axis=axis, height=heights, reset_centres=spawns)
    return dict(spawn_height_error_m=spawn_error, pyramid_face_angles_deg=slopes,
        rough_height_minmax_m=[float(rough.min()), float(rough.max())], seamless_region_boundaries=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--recipe', type=Path, help='Optional saved effective-training-config.json; no weights loaded')
    p.add_argument('--viewer', choices=('viser',))
    p.add_argument('--port', type=int, default=8770)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = FudanWalkCfg()
    if a.recipe is not None:
        for name, values in json.loads(a.recipe.read_text()).items():
            for key, value in values.items():
                setattr(getattr(cfg, name), key, value)
    cfg.terrain.mode = 'regions'
    torch.set_num_threads(1)
    report = dict(passed=False, kind='physical-visible-precheck', num_envs=20,
        terrain=record(cfg), critic_observation=critic_record(cfg), optimizer_updates=0, model_forwards=0, physics_steps=0)
    env = server = None
    started = time.time()
    try:
        report['geometry'] = geometric_checks(cfg, a.output)
        report['integration'] = integration_checks(cfg, a.recipe, a.output)
        from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
        from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.commands import command_config
        from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import posture_reward, height_reward
        from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.wheel_motor import response_record
        from scripts.agent.fudan_ppo_walk.precheck import initial_geometry
        with patch('torch.optim.Adam.step', side_effect=AssertionError('No optimizer update')), \
             patch('torch.nn.Linear.forward', side_effect=AssertionError('No network forward')):
            env = FudanEnv(20, seed=30001, randomized=True, cfg=cfg)
            ids = torch.arange(20, device=env.device)
            env.reset_idx(ids)
            sim = env.env.sim
            sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()
            assert env.env._region_kind.bincount().tolist() == [10, 5, 5]
            assert not env.terrain_invalid.any()
            if cfg.observations.critic_height_scan:
                from .check_critic_scan import check
                report['critic_scan'] = check(env, a.output)
                env.reset_idx(ids)
                sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()
            initial = initial_geometry(env, [{'id': f'region_{i}'} for i in range(20)])
            for i, row in enumerate(initial):
                row['clearance_above_spawn_ground_m'] = row['minimum_collision_mesh_height_m']-float(env.env.scene.env_origins[i, 2])
                assert row['clearance_above_spawn_ground_m'] >= -.0001, row
            poses = sim.data.qpos.clone()
            env.reset_idx(ids[[0, 11, 17]])
            keep = torch.ones(20, device=env.device, dtype=torch.bool)
            keep[[0, 11, 17]] = False
            assert torch.equal(poses[keep], sim.data.qpos[keep])
            # 将测量点移到真实坡面/崎岖处，只更新几何，不施加动力或当作行驶成功。
            saved = sim.data.qpos.clone()
            ground_checks = []
            for point in ((3., 4.), (4., 3.), (3., 12.), (10.3, -6.2), (-2., 1.)):
                sim.data.qpos[0, :3] = torch.tensor([*point, 1.2], device=env.device)
                sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()
                ray_height = 1.2-float(env.height[0])
                data = mujoco.MjData(sim.mj_model)
                mujoco.mj_forward(sim.mj_model, data)
                distance = mujoco.mj_ray(sim.mj_model, data, np.array([*point, 2.]),
                    np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, None)
                actual = 2.-distance
                assert abs(ray_height-actual) < 2e-5, (point, ray_height, actual)
                ground_checks.append(dict(xy=point, ground_height_m=ray_height, clearance_m=float(env.height[0])))
            sim.data.qpos.copy_(saved)
            sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()
            # 越界和射线缺测只能终止受影响回合，不能给出虚假的平地奖励。
            sim.data.qpos[0, 0] = cfg.terrain.size/2-.5
            sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()
            with patch.object(env.env, 'step', return_value=None):
                _, _, reward, done, info, _ = env.step(torch.zeros(20, 6, device=env.device), auto_reset=False, resample=False)
            assert done[0] and reward[0] == 0 and not info['time_outs'][0]
            env.reset_idx(ids)
            env.env.scene['walk_ground'].data.distances[0, 0] = -1.
            with patch.object(env.env, 'step', return_value=None):
                _, _, reward, done, info, _ = env.step(torch.zeros(20, 6, device=env.device), auto_reset=False, resample=False)
            assert done[0] and reward[0] == 0 and not info['time_outs'][0]
            env.reset_idx(ids)
            # 在真实坡面/粗糙处做短时落地接触诊断，避免只检查平顶上的接触。
            probes = [(2., 4.), (10., 4.), (2., 12.), (10., 12.),
                      (5.5, -12.), (13.5, -12.), (5.5, -4.), (13.5, -4.)]
            probe_data = mujoco.MjData(sim.mj_model)
            mujoco.mj_forward(sim.mj_model, probe_data)
            for i, point in enumerate(probes):
                distance = mujoco.mj_ray(sim.mj_model, probe_data, np.array([*point, 2.]),
                    np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, None)
                sim.data.qpos[i, :3] = sim.data.qpos.new_tensor([*point, 2.-distance+float(env.action.reset_root_height[i])+.05])
                sim.data.qvel[i] = 0.
            sim.forward(); env.env.scene.update(dt=0.); sim.sense()
            env.previous_position.copy_(env.robot.data.root_link_pos_w)
            probe_contacts = np.zeros(8, dtype=int)
            for _ in range(50):
                env.step(torch.zeros(20, 6, device=env.device), auto_reset=False, resample=False)
                report['physics_steps'] += cfg.control.decimation
                assert not sim.wp_data.overflow.numpy().any(), 'GPU contact/constraint capacity overflow'
                count = int(sim.wp_data.nacon.numpy()[0])
                pairs = sim.wp_data.contact.geom.numpy()[:count]
                mask = np.isin(pairs, env.env._terrain_friction_ids.cpu().numpy()).any(-1)
                worlds = sim.wp_data.contact.worldid.numpy()[:count][mask]
                probe_contacts += np.bincount(worlds, minlength=20)[:8]
            assert (probe_contacts > 0).all(), probe_contacts
            env.reset_idx(ids)
            if a.viewer:
                import viser
                from mjlab.viewer.viser.scene import MjlabViserScene
                server = viser.ViserServer(host='127.0.0.1', port=a.port, label='Pyramid / rough terrain precheck')
                scene = MjlabViserScene(server, sim.mj_model, 20, sim_model=sim.model, expanded_fields=sim.expanded_fields)
                scene.show_only_selected = False
                scene.create_scene_gui(camera_distance=35., show_debug_viz_control=False)
                scene.update(sim.data)
                print(f'20-world precheck: http://127.0.0.1:{server.get_port()}', flush=True)
            fixture = a.output/'fixtures'
            fixture.mkdir()
            mujoco.mj_saveModel(sim.mj_model, str(fixture/'model.mjb'))
            np.savez_compressed(fixture/'initial.npz', qpos=sim.data.qpos.cpu().numpy(),
                origins=env.env.scene.env_origins.cpu().numpy(), kinds=env.env._region_kind.cpu().numpy())
            contact_observations = 0
            coefficient_error = 0.
            for tick in range(200):
                env.step(torch.zeros(20, 6, device=env.device))
                report['physics_steps'] += cfg.control.decimation
                assert not sim.wp_data.overflow.numpy().any(), 'GPU contact/constraint capacity overflow'
                assert not env.terrain_invalid.any()
                count = int(sim.wp_data.nacon.numpy()[0])
                if count:
                    pairs = sim.wp_data.contact.geom.numpy()[:count]
                    mask = np.isin(pairs, env.env._terrain_friction_ids.cpu().numpy()).any(-1)
                    contact_observations += int(mask.sum())
                    if cfg.domain_rand.ground_friction_range is not None and mask.any():
                        worlds = sim.wp_data.contact.worldid.numpy()[:count][mask]
                        expected = env.env._fudan_parameters['friction'].cpu().numpy()[worlds]
                        actual = sim.wp_data.contact.friction.numpy()[:count, 0][mask]
                        coefficient_error = max(coefficient_error, float(np.abs(actual-expected).max()))
                if server is not None and tick%3 == 0:
                    scene.update(sim.data)
                    time.sleep(.01)
            assert contact_observations and coefficient_error < 1e-6
            assert torch.isfinite(sim.data.qpos).all() and torch.isfinite(sim.data.qvel).all()
            obs, critic = env.observations()
            assert obs.shape == (20, 25) and critic.shape == (20, 141) and env.history.shape == (20, 125)
            report.update(passed=True, reset_contract=env.reset_contract,
                disturbances=env.disturbance_config.record(), command_sampling=command_config(cfg),
                posture_reward=posture_reward.from_task_config(cfg), height_reward=height_reward.from_task_config(cfg),
                wheel_response=response_record(cfg), control=section_dict(cfg.control), control_timing=timing_record(cfg),
                validation_backend='mujoco_warp_gpu', initial=initial, region_counts=[10, 5, 5],
                local_height_ground_probes=ground_checks, partial_reset_preserves_other_worlds=True,
                boundary_failure_reward_zero=True, contact_observations=contact_observations,
                maximum_contact_coefficient_error=coefficient_error,
                ground_friction_by_env=env.env._fudan_parameters['friction'].cpu().tolist(),
                slope_and_rough_contact_probes=probe_contacts.tolist(), missing_ray_failure_reward_zero=True,
                capacity_overflow=False,
                actual_config={k: section_dict(v) for k, v in vars(cfg).items() if hasattr(v, '__dict__')},
                recipe_source=str(a.recipe), recipe_sha256=hashlib.sha256(a.recipe.read_bytes()).hexdigest() if a.recipe else None,
                scope='Geometry, reset, local height, real GPU contacts and finite zero-action dynamics; not terrain traversal acceptance')
            render_preview(a.output, cfg.terrain)
    except BaseException as exc:
        report['passed'] = False
        report['error'] = repr(exc)
        raise
    finally:
        report['wall_s'] = time.time()-started
        paths = [Path(__file__), Path('mjlab_training/src/ct_mjlab/envs/chuanliantui/fudan_ppo_walk/terrain.py'),
                 Path('mjlab_training/src/ct_mjlab/envs/chuanliantui/fudan_ppo_walk/fudan_ppo_walk.py'),
                 Path('mjlab_training/src/ct_mjlab/envs/chuanliantui/fudan_ppo_walk/fudan_ppo_walk_config.py'),
                 Path('mjlab_training/src/ct_mjlab/scripts/train.py'),
                 Path('mjlab_training/src/ct_mjlab/rsl_rl/runners/fudan_on_policy_runner.py')]
        report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if server is not None:
            server.stop()
        if env is not None:
            env.close()
    print(json.dumps({k: report[k] for k in ('passed', 'region_counts', 'physics_steps', 'optimizer_updates')}, indent=2))


if __name__ == '__main__':
    main()
