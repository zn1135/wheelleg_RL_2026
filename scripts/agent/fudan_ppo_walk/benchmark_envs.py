"""Time real sampling and PPO updates in disposable copies of a saved policy."""
import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import mujoco
import torch

from ct_mjlab.fudan_ppo_walk.contract import CONTRACT, RESET_CONTRACT, WEIGHTS
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner, initialize_rollout


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = json.loads(args.source_manifest.read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name

    report = dict(passed=False, num_envs=args.num_envs, seed=19001,
                  warmup_iterations=2, measured_iterations=8, iterations=[],
                  primary_updates=0, encoder_updates=0, policy_steps=0,
                  disposable_parameters=True, saves_policy=False)
    env = None
    runner = None
    samples = []
    phase = 'initialization'
    monitor = subprocess.Popen(
        ['nvidia-smi', '--id=0', '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
         '--format=csv,noheader,nounits', '--loop-ms=250'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def read_resources():
        for line in monitor.stdout:
            try:
                values = [float(value.strip()) for value in line.split(',')]
                samples.append(dict(time_unix=time.time(), phase=phase, memory_MiB=values[0],
                                    gpu_utilization_percent=values[1], temperature_C=values[2],
                                    power_W=values[3], sm_clock_MHz=values[4]))
            except ValueError:
                continue

    thread = threading.Thread(target=read_resources, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        if time.time() >= args.deadline:
            raise TimeoutError('Benchmark budget expired before initialization')
        torch.manual_seed(19001)
        env = FudanEnv(args.num_envs, seed=19001, randomized=True)
        runner = FudanRunner(env, log_dir=args.output / 'events')
        runner.deadline = args.deadline
        saved = torch.load(args.checkpoint, map_location='cuda:0', weights_only=False)
        assert saved['contract'] == CONTRACT and saved['reset_contract'] == RESET_CONTRACT
        runner.policy.load_state_dict(saved['model_state_dict'])
        runner.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
        runner.alg.extra_optimizer.load_state_dict(saved['encoder_optimizer_state_dict'])
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
        report['initial_optimizer_step'] = int(next(iter(runner.alg.optimizer.state.values()))['step'])
        del saved
        obs, critic, history = initialize_rollout(env)
        report['policy_steps'] = 1
        report['initialization_s'] = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        reward_names = tuple(WEIGHTS)

        for iteration in range(10):
            if time.time() >= args.deadline:
                raise TimeoutError('Benchmark budget expired')
            phase = 'warmup' if iteration < 2 else 'measured'
            torch.cuda.synchronize()
            tick = time.perf_counter()
            reward_sum = obs.new_zeros(())
            term_sums = obs.new_zeros(len(reward_names))
            # 与正式runner相同：48步真实采样，计算回报，再各做20次Adam。
            with torch.inference_mode():
                for _ in range(48):
                    if time.time() >= args.deadline:
                        raise TimeoutError('Benchmark budget expired during collection')
                    actions = runner.alg.act(obs, history, critic)
                    obs, critic, reward, done, info, history = env.step(actions)
                    runner.alg.process_env_step(reward, done, info, obs)
                    reward_sum += reward.mean()
                    term_sums += torch.stack([info['terms'][name] for name in reward_names]).mean(dim=1)
                    report['policy_steps'] += 1
                runner.alg.compute_returns(torch.cat((critic, runner.policy.encode(history)), -1))
            torch.cuda.synchronize()
            collected = time.perf_counter()
            losses = runner.alg.update()
            runner.iteration += 1
            runner.writer.add_scalar('Loss/value', losses[0], runner.iteration)
            runner.writer.add_scalar('Loss/policy', losses[1], runner.iteration)
            runner.writer.add_scalar('Loss/velocity_estimator', losses[3], runner.iteration)
            runner.writer.add_scalar('Train/reward', (reward_sum / 48).item(), runner.iteration)
            for name, value in zip(reward_names, (term_sums / 48).tolist()):
                runner.writer.add_scalar(f'Rewards/{name}', value, runner.iteration)
            runner.writer.add_scalar('Train/speed_limit', env.speed_limit, runner.iteration)
            runner.writer.add_scalar('Train/learning_rate', runner.alg.learning_rate, runner.iteration)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            row = dict(iteration=iteration + 1, measured=iteration >= 2,
                       collection_s=collected - tick, update_and_logging_s=finished - collected,
                       total_s=finished - tick, transitions=args.num_envs * 48)
            report['iterations'].append(row)
            print(json.dumps(row), flush=True)

        assert runner.primary_updates == runner.encoder_updates == 200
        assert report['policy_steps'] == 481
        assert all(torch.isfinite(value).all() for value in runner.policy.state_dict().values())
        measured = [row for row in report['iterations'] if row['measured']]
        wall = sum(row['total_s'] for row in measured)
        report.update(passed=True, measured_wall_s=wall,
                      measured_transitions=args.num_envs * 48 * 8,
                      transitions_per_second=args.num_envs * 48 * 8 / wall,
                      mean_collection_s=sum(row['collection_s'] for row in measured) / 8,
                      mean_update_and_logging_s=sum(row['update_and_logging_s'] for row in measured) / 8,
                      mean_iteration_s=wall / 8,
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        phase = 'cleanup'
        report['case_wall_s'] = time.perf_counter() - started
        if runner is not None:
            report['primary_updates'] = runner.primary_updates
            report['encoder_updates'] = runner.encoder_updates
            runner.writer.flush()
            runner.writer.close()
        if env is not None:
            env.close()
        monitor.terminate()
        monitor.wait(timeout=5)
        thread.join(timeout=2)
        measured_resources = [sample for sample in samples if sample['phase'] == 'measured']
        report['sampled_gpu_total_peak_MiB'] = max((sample['memory_MiB'] for sample in samples), default=None)
        report['sampled_measured_gpu_peak_MiB'] = max((sample['memory_MiB'] for sample in measured_resources), default=None)
        report['sampled_measured_temperature_peak_C'] = max((sample['temperature_C'] for sample in measured_resources), default=None)
        report['torch_version'] = torch.__version__
        report['mujoco_version'] = mujoco.__version__
        (args.output / 'resources.json').write_text(json.dumps(samples, indent=2) + '\n')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
