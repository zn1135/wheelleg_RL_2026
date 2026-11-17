"""Authorized5000 ->8000 continuation on the selected continuous regional terrain."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from scripts.agent.fudan_ppo_walk.resume_pd200_random_from3000 import ROOT, PYTHON, XML, sha, write, same
from scripts.agent.fudan_ppo_walk.run_torque_ablation import run_command

MODULE = 'scripts.agent.fudan_ppo_walk.resume_regions3000'
SOURCE_RUN = ROOT/'logs/mjlab/fudan_ppo_walk/20261116_high_yaw10_resume4000_to5000_01'
SOURCE = SOURCE_RUN/'model_final.pt'
SOURCE_SHA = '010106ac0b2ee356a630ab27ebe1c85ab2a6c843d21a1b4362ab81f1aafe0222'
LOG = ROOT/'logs/mjlab/fudan_ppo_walk/20261116_regions10to20_resume5000_to8000_01'


def prepare(out):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, section_dict
    from ct_mjlab.envs.fudan_ppo_walk.terrain import record
    assert sha(SOURCE) == SOURCE_SHA
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert source['iterations'] == 5000 and not source['partial_update']
    assert source['primary_updates'] == source['encoder_updates'] == 100000
    for key in ('optimizer_state_dict', 'encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in source[key]['state'].values()} == {100000}
    config = json.loads((SOURCE_RUN/'effective-training-config.json').read_text())
    task = FudanWalkCfg()
    task.terrain.mode = 'regions'
    assert task.terrain.slope_degrees == [10., 20.]
    config['terrain'] = section_dict(task.terrain)
    proof_path = ROOT/'artifacts/fudan_ppo_walk/20261116-regional-pyramid10to20-01/report.json'
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and proof['terrain'] == record(task)
    assert all(proof['actual_config'][key] == value for key, value in config.items())
    for name, digest in proof['source_sha256'].items():
        path = ROOT/name
        if path.name == 'fudan_ppo_walk_config.py':
            import hashlib
            prior = path.read_text().replace('velocity_max = 3.', 'velocity_max = 2.5')
            assert hashlib.sha256(prior.encode()).hexdigest() == digest, name
        else:
            assert sha(path) == digest, name
    config['commands']['velocity_max'] = 3.
    assert not LOG.exists()
    out.mkdir(parents=True, exist_ok=False)
    budget = dict(authorization='User requested start3000 iterations after selecting pyramid slopes10-20; continue latest5000 to8000',
        checkpoint=str(SOURCE), checkpoint_sha256=SOURCE_SHA, source_run=str(SOURCE_RUN),
        source_config=str(SOURCE_RUN/'effective-training-config.json'),
        start_iteration=5000, target_iteration=8000, additional_iterations=3000,
        num_envs=4096, rollout_steps=48, seed=1, log_dir=str(LOG), actual_config=config,
        terrain=proof['terrain'], certified_terrain_precheck=str(proof_path),
        # 只缩减预分配接触/约束缓冲；20及4096实际容量检查必须均无溢出。
        runtime_capacity=dict(nconmax=64, njmax=256),
        control_timing=source['control_timing'], disturbances=source['disturbances'],
        command_sampling=source['command_sampling'], height_reward=source['height_reward'],
        physical_wheel_model=source['physical_wheel_model'], xml_sha256=sha(XML),
        inherited_abandoned_rollout_transitions=source['transitions']-5000*4096*48,
        expected_added_transitions=3000*4096*48, expected_added_optimizer_steps=60000,
        created_unix=time.time())
    assert budget['inherited_abandoned_rollout_transitions'] == 184320
    write(out/'budget.json', budget)
    write(out/'target-effective-config.json', config)
    shutil.copy2(__file__, out/'launcher.py')
    shutil.copy2(proof_path, out/'previous-terrain-precheck.json')
    write(out/'source-identity.json', dict(branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), dirty=True))
    write(out/'status.json', dict(status='prepared', training_started=False))


def worker(out, stage):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, timing_record
    from ct_mjlab.envs.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.envs.fudan_ppo_walk.terrain import record
    from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
    b = json.loads((out/'budget.json').read_text())
    assert sha(SOURCE) == b['checkpoint_sha256'] and sha(XML) == b['xml_sha256']
    for name, digest in b.get('approved_source_sha256', {}).items():
        assert sha(ROOT/name) == digest, name
    torch.set_num_threads(1)
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    old_cfg, old_init = physical.make_env_cfg, physical.FudanEnv.__init__
    stats = dict(physics_steps=0, constraint_peak=0, global_contact_peak=0,
                 capacity_overflow=False, pd_updates=0, held_ticks=0,
                 held_torque_exact=True, wrench_max_error=0., push_observations=0)

    def make_cfg(*args, **kwargs):
        cfg = old_cfg(*args, **kwargs)
        cfg.sim.nconmax = b['runtime_capacity']['nconmax']
        cfg.sim.njmax = b['runtime_capacity']['njmax']
        return cfg

    def checked_init(env, *args, **kwargs):
        old_init(env, *args, **kwargs)
        sim, action = env.env.sim, env.action
        assert record(env.cfg) == b['terrain'] and timing_record(env.cfg) == b['control_timing']
        if 'critic_observation' in b:
            from ct_mjlab.envs.fudan_ppo_walk.terrain import critic_record
            assert critic_record(env.cfg) == b['critic_observation']
        assert env.disturbance_config.record() == b['disturbances']
        assert env.command_sampling == b['command_sampling'] and env.cfg.commands.velocity_max == 3.
        assert env.wheel_physics_profile == b['physical_wheel_model']
        assert (action.cfg.leg_stiffness, action.cfg.leg_damping) == (20., 1.)
        assert sim.wp_data.njmax == 256 and sim.wp_data.naconmax == env.num_envs*64
        for attr, key in (('kp_scale', 'kp_range'), ('kd_scale', 'kd_range'),
                          ('motor_strength', 'motor_strength_range'), ('nominal_offset', 'nominal_offset_range')):
            v = getattr(action, attr)
            low, high = b['actual_config']['domain_rand'][key]
            assert v.min() >= low-1e-7 and v.max() <= high+1e-7, attr
        state = action.disturbances
        assert state.episode_wrench[:, :3].abs().le(3.).all() and state.episode_wrench[:, 3:].eq(0).all()
        old_resample = env.resample
        def resample(ids):
            old_resample(ids)
            high = ids[env.commands[ids, 1].abs() > 2.]
            assert env.commands[high, 0].eq(0).all()
            if len(ids) and not (out/(stage+'-commands.json')).exists():
                write(out/(stage+'-commands.json'), dict(num_envs=len(ids), high_yaw_count=len(high), high_yaw_vx_zero=True))
        env.resample = resample
        step = sim.step
        if stage == 'precheck':
            force, apply = state.force, action.apply_actions
            pulse = [torch.zeros(env.num_envs, 3, device=env.device)]
            def observed_force(quat, scale):
                pulse[0] = force(quat, scale)
                stats['push_observations'] += int(pulse[0].ne(0).any(-1).sum())
                return pulse[0]
            state.force = observed_force
            def observed_apply():
                if not stats['physics_steps']:
                    state.ticks.fill_(state.push_interval-1)
                tick, before = action._pd_tick, action.torques.clone()
                apply()
                stats['pd_updates'] += int(tick == 0)
                stats['held_ticks'] += int(tick != 0)
                if tick != 0:
                    stats['held_torque_exact'] &= bool(torch.equal(before, action.torques))
            action.apply_actions = observed_apply
        def observed_step(*args, **kwargs):
            if stage == 'precheck':
                expected = state.wrench(sim.data.qpos[:, 3:7])
                expected[:, :3] += pulse[0]
                stats['wrench_max_error'] = max(stats['wrench_max_error'], float((sim.data.xfrc_applied[:, action._base_id]-expected).abs().max()))
            result = step(*args, **kwargs)
            stats['physics_steps'] += 1
            stats['constraint_peak'] = max(stats['constraint_peak'], int(sim.wp_data.nefc.numpy().max()))
            stats['global_contact_peak'] = max(stats['global_contact_peak'], int(sim.wp_data.nacon.numpy()[0]))
            stats['capacity_overflow'] |= bool(sim.wp_data.overflow.numpy().any())
            assert not stats['capacity_overflow'], 'GPU capacity overflow'
            return result
        # 训练阶段用环境自身溢出守卫，避免逐物理步CPU统计影响正式吞吐。
        if stage != 'train':
            sim.step = observed_step
        write(out/(stage+'-actual-configuration.json'), dict(num_envs=env.num_envs, terrain=record(env.cfg),
            capacity=b['runtime_capacity'], control_timing=timing_record(env.cfg), disturbances=env.disturbance_config.record()))

    with ExitStack() as stack:
        for section, values in b['actual_config'].items():
            for key, value in values.items():
                # 保留类级默认平地；预检实例和训练--terrain各自显式选分区。
                if section == 'terrain' and key == 'mode' or section == 'observations':
                    continue
                stack.enter_context(patch.object(getattr(FudanWalkCfg, section), key, value, create=True))
        stack.enter_context(patch.object(physical, 'make_env_cfg', make_cfg))
        stack.enter_context(patch.object(physical.FudanEnv, '__init__', checked_init))
        if stage == 'precheck':
            from scripts.agent.fudan_ppo_walk import check_region_terrain
            with patch.object(sys, 'argv', ['precheck', '--output', str(out/'precheck'),
            '--recipe', str(out/'target-effective-config.json'), '--viewer', 'viser', '--port', '8806']):
                check_region_terrain.main()
            assert stats['physics_steps'] == 2500 and stats['held_ticks'] == 4*stats['pd_updates']
            assert stats['held_torque_exact'] and stats['wrench_max_error'] < 1e-6 and stats['push_observations'] >= 20
            write(out/'precheck-runtime-audit.json', dict(passed=True, **stats))
        elif stage == 'capacity':
            capacity(out, b, source, stats)
        else:
            from ct_mjlab.scripts import train
            old_save, old_snapshot = runner.FudanRunner.save, runner.snapshot
            def save(learner, name):
                path = old_save(learner, name)
                if name == 'model_initial.pt':
                    initial = torch.load(path, map_location='cpu', weights_only=False)
                    for key in ('model_state_dict', 'optimizer_state_dict', 'encoder_optimizer_state_dict', 'torch_rng',
                                'cuda_rng', 'iterations', 'transitions', 'primary_updates', 'encoder_updates', 'speed_limit', 'policy_steps'):
                        assert same(initial[key], source[key]), key
                    assert initial['terrain'] == b['terrain']
                    if 'critic_observation' in b:
                        assert initial['critic_observation'] == b['critic_observation']
                    write(out/'initialization-audit.json', dict(passed=True, source_checkpoint_sha256=sha(SOURCE),
                        all_networks_exact=True, both_optimizers_exact=True, rng_and_counters_exact=True,
                        terrain_changed=True, physics_and_history_reinitialized=True))
                return path
            def snapshot(path):
                old_snapshot(path)
                path = Path(path)
                shutil.copy2(out/'launcher.py', path/'continuation-launcher.py')
                write(path/'effective-training-config.json', b['actual_config'])
                write(path/'runtime-capacity.json', b['runtime_capacity'])
                shutil.copy2(out/'budget.json', path/'authorized-budget.json')
                shutil.copy2(XML, path/'source/resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')
            with patch.object(runner, 'snapshot', snapshot), patch.object(runner.FudanRunner, 'save', save):
                scan = ['--critic-height-scan'] if b['actual_config'].get('observations', {}).get('critic_height_scan') else []
                train.main(['--task', 'fudan_ppo_walk', '--terrain', 'regions', *scan, '--num-envs', '4096',
                    '--iterations', str(b['additional_iterations']), '--seed', '1', '--resume', str(SOURCE),
                    '--precheck', str(out/'precheck/report.json'), '--log-dir', b['log_dir']])


def capacity(out, b, source, stats):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner
    cfg = FudanWalkCfg()
    cfg.terrain.mode = 'regions'
    cfg.observations.critic_height_scan = b['actual_config'].get('observations', {}).get('critic_height_scan', False)
    env = learner = None
    try:
        with patch('torch.optim.Adam.step', side_effect=AssertionError('Capacity preflight cannot update')):
            env = FudanEnv(4096, seed=1, cfg=cfg)
            learner = FudanRunner(env, log_dir=out/'capacity-memory-only')
            learner.load(SOURCE, allow_terrain_change=True,
                allow_critic_observation_change=b['actual_config'].get('observations', {}).get('critic_height_scan', False))
            inherited_cap = env.speed_limit
            env.speed_limit = 3.
            env.resample(torch.arange(4096, device=env.device))
            sampled_max = float(env.commands[:, 0].abs().max())
            assert sampled_max > 2.5 and sampled_max <= 3.
            env.speed_limit = inherited_cap
            size = 4096*48//4
            before = {k: v.detach().cpu().clone() for k, v in learner.policy.state_dict().items()}
            inputs, history, critic = (torch.zeros(size, n, device='cuda:0') for n in (25, 125, 144))
            action = learner.policy.act(inputs, history)
            loss = learner.policy.evaluate(critic).square().mean()-.001*learner.policy.get_actions_log_prob(action).mean()
            loss.backward(); learner.policy.zero_grad(set_to_none=True)
            learner.policy.encode(history).square().mean().backward(); learner.policy.zero_grad(set_to_none=True)
            del inputs, history, critic, action, loss
            obs, critic, history = learner.initialize_rollout()
            started = time.perf_counter()
            with torch.inference_mode():
                for _ in range(48):
                    action = learner.alg.act(obs, history, critic)
                    obs, critic, reward, done, info, history = env.step(action)
                    assert torch.isfinite(critic).all()
                    if cfg.observations.critic_height_scan:
                        assert env.terrain_samples.shape == (4096, 77)
                        assert not env.terrain_invalid.any()
                    learner.alg.process_env_step(reward, done, info, obs)
                learner.alg.compute_returns(torch.cat((critic, learner.policy.encode(history)), -1))
            assert all(torch.equal(v.cpu(), before[k]) for k, v in learner.policy.state_dict().items())
            assert learner.primary_updates == source['primary_updates']
            assert learner.encoder_updates == source['encoder_updates']
            assert env.env._region_kind.bincount().tolist() == [2048, 1024, 1024]
            free, total = torch.cuda.mem_get_info()
            write(out/'capacity-audit.json', dict(passed=True, num_envs=4096, region_counts=[2048, 1024, 1024],
                minibatch=size, actual_ppo_rollout_steps=48, weights_unchanged=True, optimizer_updates=0,
                cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(), free_gpu_memory_bytes=free,
                total_gpu_memory_bytes=total, sampling_transitions_per_second=4096*48/(time.perf_counter()-started),
                sampling_is_not_training_throughput=True, **stats))
            path = out/'capacity-audit.json'
            result = json.loads(path.read_text())
            result.update(inherited_speed_limit=inherited_cap, configured_velocity_max=3.,
                          sampled_absolute_vx_max_at_cap3=sampled_max)
            write(path, result)
    finally:
        if learner is not None:
            learner.writer.close()
        if env is not None:
            env.close()


def audit(out):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    b = json.loads((out/'budget.json').read_text())
    log = Path(b['log_dir'])
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    final = torch.load(log/'model_final.pt', map_location='cpu', weights_only=False)
    r = json.loads((log/'training_report.json').read_text())
    assert r['stopped_by'] == 'iteration_budget' and r['iterations_added'] == b['additional_iterations']
    assert final['iterations'] == b['target_iteration'] and not final['partial_update']
    assert final['transitions']-source['transitions'] == b['expected_added_transitions']
    expected_updates = b['target_iteration']*20
    assert final['primary_updates'] == final['encoder_updates'] == expected_updates
    assert final['terrain'] == b['terrain']
    assert json.loads((out/'initialization-audit.json').read_text())['passed']
    for key in ('control', 'control_timing', 'height_reward', 'disturbances', 'physical_wheel_model', 'command_sampling'):
        assert final[key] == source[key], key
    for key in ('optimizer_state_dict', 'encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in final[key]['state'].values()} == {expected_updates}
    assert all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
    events = EventAccumulator(str(log), size_guidance={'scalars': 0}).Reload()
    for tag in ['Train/reward', *['Rewards/'+name for name in final['reward_names']]]:
        assert [v.step for v in events.Scalars(tag)] == list(range(b['start_iteration']+1, b['target_iteration']+1)), tag
    manifest = json.loads((log/'source_sha256.json').read_text())
    drift = []
    for name, digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert sha(log/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/')) == digest, name
            if sha(ROOT/name) != digest:
                drift.append(name)
    if 'critic_observation' in b:
        assert final['critic_observation'] == b['critic_observation']
    write(out/'training-audit.json', dict(budget_and_learning_state_passed=True, final_iteration=b['target_iteration'],
        additional_iterations=b['additional_iterations'], final_sha256=sha(log/'model_final.pt'), saved_source_snapshot_passed=True,
        current_workspace_drift=drift, behavior_accepted=False))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('prepare', 'precheck', 'capacity', 'train', 'run'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if a.mode == 'prepare':
        prepare(out)
    elif a.mode != 'run':
        worker(out, a.mode)
    else:
        try:
            assert not LOG.exists()
            for stage in ('precheck', 'capacity'):
                run_command(out, stage, [str(PYTHON), '-u', '-m', MODULE, stage, '--output', str(out)])
                name = 'precheck-runtime-audit.json' if stage == 'precheck' else 'capacity-audit.json'
                assert json.loads((out/name).read_text())['passed']
            run_command(out, 'train', [str(PYTHON), '-u', '-m', MODULE, 'train', '--output', str(out)])
            audit(out)
            write(out/'status.json', dict(status='training_complete', final_iteration=8000, iterations_added=3000, behavior_accepted=False))
        except BaseException as exc:
            write(out/'status.json', dict(status='failed', error=repr(exc), automatic_retry=False))
            raise


if __name__ == '__main__':
    main()
