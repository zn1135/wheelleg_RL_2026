"""实际GPU射线→Critic检查；CPU mj_ray只用于独立几何复算，不推进CPU物理。"""
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch


def check(env, output):
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.terrain import GroundRay, critic_record, LEGACY_CRITIC
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfgPPO
    from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner
    from ct_mjlab.envs import task_registry
    from ct_mjlab.scripts.train import main as train_main
    sim = env.env.sim
    poses = sim.data.qpos.clone()
    cpu = mujoco.MjData(sim.mj_model)
    mujoco.mj_forward(sim.mj_model, cpu)
    origins, _ = GroundRay(size=(1., .6), resolution=.1).generate_rays(device=env.device)
    assert origins.shape == (77, 3)
    np.testing.assert_allclose(origins[:11, 0].cpu(), np.linspace(-.5, .5, 11), atol=1e-7)
    rows = []
    max_error = 0.
    try:
        for x, y, z, yaw in ((-8., 0., 1.2, 0.), (2., 4., 1.2, 0.),
                              (2., 4., 1.2, math.pi/2), (2.2, 4.1, 1.2, .7),
                              (10.3, -6.2, 1.2, .7)):
            sim.data.qpos[0, :3] = poses.new_tensor([x, y, z])
            sim.data.qpos[0, 3:7] = poses.new_tensor([math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
            sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()
            assert not env.terrain_invalid[0]
            local = origins.cpu().numpy()
            world = local.copy()
            world[:, 0] = x+math.cos(yaw)*local[:, 0]-math.sin(yaw)*local[:, 1]
            world[:, 1] = y+math.sin(yaw)*local[:, 0]+math.cos(yaw)*local[:, 1]
            world[:, 2] += z
            expected = []
            for origin in world:
                distance = mujoco.mj_ray(sim.mj_model, cpu, origin.astype(np.float64),
                    np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, None)
                assert distance > 0
                expected.append(origin[2]-distance)
            actual = env.terrain_samples[0].cpu().numpy()
            error = float(np.max(np.abs(actual-expected)))
            assert error < 2e-5, (x, y, yaw, error)
            max_error = max(max_error, error)
            rng = torch.cuda.get_rng_state_all()
            actor, critic = env.observations()
            correct = ((env.position[:, 2, None]-env.terrain_samples-env.cfg.normalization.height_center)
                       .clamp(-1., 1.)*env.cfg.normalization.height_scale)
            assert torch.equal(critic[:, 46:123], correct)
            env.cfg.observations.critic_height_scan = False
            torch.cuda.set_rng_state_all(rng)
            old_actor, old_critic = env.observations()
            env.cfg.observations.critic_height_scan = True
            assert torch.equal(actor, old_actor)
            assert torch.equal(critic[:, :46], old_critic[:, :46])
            assert torch.equal(critic[:, 123:], old_critic[:, 123:])
            spread = float(np.ptp(actual))
            if x < 0:
                assert spread < 1e-6 and torch.equal(critic[0], old_critic[0])
            else:
                assert spread > .001
                assert not torch.equal(critic[0, 46:123], old_critic[0, 46:123])
            rows.append(dict(position=[x, y, z], yaw=yaw, height_spread_m=spread, max_ray_error_m=error))
        sim.data.qpos.copy_(poses)
        sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()
        env.env.scene['walk_scan'].data.distances[0, 0] = -1.
        with patch.object(env.env, 'step', return_value=None):
            obs, critic, reward, done, info, _ = env.step(torch.zeros(env.num_envs, 6, device=env.device),
                                                       auto_reset=False, resample=False)
        assert done[0] and reward[0] == 0 and not info['time_outs'][0]
        assert torch.isfinite(obs).all() and torch.isfinite(critic).all()
    finally:
        env.cfg.observations.critic_height_scan = True
        sim.data.qpos.copy_(poses)
        sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()

    # 相同物理地形也不能用旧重复高度预检批准新输入语义。
    stale = output/'legacy-critic-precheck.json'
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.terrain import record
    stale.write_text(json.dumps(dict(terrain=record(env.cfg), critic_observation=LEGACY_CRITIC)))
    with patch.object(task_registry, 'make_env', side_effect=AssertionError('Reject before GPU construction')):
        try:
            train_main(['--task', 'fudan_ppo_walk', '--terrain', 'regions', '--critic-height-scan',
                        '--num-envs', '20', '--iterations', '1', '--precheck', str(stale),
                        '--log-dir', str(output/'must-not-train-scan')])
        except ValueError as exc:
            assert 'precheck critic observation differs' in str(exc)
        else:
            raise AssertionError('Legacy critic precheck accepted')
    source = output/'legacy-critic-loader.pt'
    torch.save(dict(terrain=record(env.cfg)), source)
    dummy = SimpleNamespace(env=SimpleNamespace(cfg=env.cfg), train_cfg=FudanWalkCfgPPO(), device='cpu')
    try:
        FudanRunner.load(dummy, source)
    except ValueError as exc:
        assert 'Critic observation differs' in str(exc)
    else:
        raise AssertionError('Critic migration accepted implicitly')
    result = dict(passed=True, critic_observation=critic_record(env.cfg), probes=rows,
        max_ray_error_m=max_error, actor_exact_with_same_rng=True, other_critic_slots_exact=True,
        flat_legacy_exact=True, missing_scan_terminates_zero_reward_no_bootstrap=True,
        stale_precheck_rejected=True, implicit_checkpoint_migration_rejected=True,
        optimizer_updates=0, model_forwards=0, cpu_physics_steps=0)
    (output/'critic-scan-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
