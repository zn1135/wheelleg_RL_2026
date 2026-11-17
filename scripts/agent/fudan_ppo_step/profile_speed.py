"""Local isolated sampling/gradient timing; Adam updates are forbidden."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--rollouts', type=int, default=3)
    parser.add_argument('--whole-only', action='store_true', help='Time whole rollouts without per-stage instrumentation')
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg, FudanStepCfgPPO
    from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_step.runner import StepRunner
    from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner

    torch.set_num_threads(1)
    env = runner = None
    report = dict(optimizer_updates=0, passed=False, num_envs=args.num_envs,
                  gpu=torch.cuda.get_device_name(0), whole_only=args.whole_only, stages={}, rollouts=[])
    calls = defaultdict(int)
    host = defaultdict(float)
    events = defaultdict(list)

    def timed(name, function):
        if args.whole_only: return function
        def measured(*a, **kw):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tick = time.perf_counter(); start.record()
            result = function(*a, **kw)
            end.record(); host[name] += time.perf_counter() - tick
            events[name].append((start, end)); calls[name] += 1
            return result
        return measured

    def no_update(_runner, _optimizer, counter):
        calls['blocked_' + counter] += 1

    try:
        with patch('torch.optim.Adam.step', side_effect=AssertionError('Profiling cannot train')):
            env = StepEnv(args.num_envs, seed=1, cfg=FudanStepCfg(), randomized=True)
            runner = StepRunner(env, train_cfg=FudanStepCfgPPO(), log_dir=args.output/'runner', device='cuda:0')
            runner.load(args.checkpoint)
            frozen = {key: value.detach().cpu().clone() for key, value in runner.policy.state_dict().items()}
            optimizer_steps = [{key: float(value['step']) for key, value in opt.state_dict()['state'].items()}
                               for opt in (runner.alg.optimizer, runner.alg.extra_optimizer)]
            obs, critic, history = env.reset()
            # 热身不写PPO缓存，随后从同一恢复状态重新开始。
            with torch.inference_mode():
                for _ in range(10):
                    action, _ = runner.policy.act_inference(obs, history)
                    obs, critic, _, _, _, history = env.step(action)
            runner.load(args.checkpoint)
            with torch.inference_mode():
                obs, critic, history = env.reset()
            torch.cuda.synchronize()
            env.contacts.update = timed('contact_statistics', env.contacts.update)
            env.env.sim.step = timed('physics', env.env.sim.step)
            env.env.scene.update = timed('scene_and_sensors', env.env.scene.update)
            env._progress = timed('progress', env._progress)
            env.reset_idx = timed('resets', env.reset_idx)
            env.observations = timed('observations', env.observations)
            if hasattr(env, '_record_episodes'):
                env._record_episodes = timed('episode_logging', env._record_episodes)
            base_step = timed('base_environment', FudanEnv.step)
            with patch.object(FudanEnv, 'step', base_step), patch.object(FudanRunner, 'optimizer_step', no_update):
                for index in range(args.rollouts):
                    if time.time() >= args.deadline: raise TimeoutError('Profiling deadline')
                    torch.cuda.synchronize(); began = time.perf_counter()
                    with torch.inference_mode():
                        for _ in range(runner.num_steps_per_env):
                            action = timed('policy_sampling', runner.alg.act)(obs, history, critic)
                            obs, critic, reward, done, info, history = timed('environment', env.step)(action)
                            runner.alg.process_env_step(reward, done, info, obs)
                        runner.alg.compute_returns(torch.cat((critic, runner.policy.encode(history)), -1))
                    torch.cuda.synchronize(); sampled = time.perf_counter()
                    losses = runner.alg.update()
                    torch.cuda.synchronize(); updated = time.perf_counter()
                    log_tick = time.perf_counter(); runner.log_task_metrics(); runner.writer.flush()
                    report['rollouts'].append(dict(index=index, sample_s=sampled-began,
                        backward_no_adam_s=updated-sampled, log_s=time.perf_counter()-log_tick,
                        losses=[float(v) for v in losses]))
            torch.cuda.synchronize()
            assert all(torch.equal(value.cpu(), frozen[key]) for key, value in runner.policy.state_dict().items())
            assert optimizer_steps == [{key: float(value['step']) for key, value in opt.state_dict()['state'].items()}
                                       for opt in (runner.alg.optimizer, runner.alg.extra_optimizer)]
            report['stages'] = {name: dict(calls=calls[name], host_s=host[name],
                gpu_timeline_s=sum(start.elapsed_time(end) for start, end in pairs)/1000.)
                for name, pairs in events.items()}
            report.update(passed=True, all_weights_unchanged=True, optimizer_steps_unchanged=True,
                blocked_primary=calls['blocked_primary_updates'], blocked_encoder=calls['blocked_encoder_updates'],
                constraint_peak=int(env.contacts.constraint_peak), capacity_overflow=bool(env.contacts.capacity_overflow.any()),
                tensors={key:digest(value) for key,value in dict(qpos=env.env.sim.data.qpos,
                    qvel=env.env.sim.data.qvel,obs=obs,critic=critic,history=history).items()},
                gradients={key:digest(p.grad) for key,p in runner.policy.named_parameters() if p.grad is not None},
                scope='Local instrumented sampling and full PPO/anchor backward with all Adam calls blocked; not real training throughput.')
            assert not report['capacity_overflow']
    except BaseException as exc:
        report['error'] = repr(exc); raise
    finally:
        if runner is not None: runner.writer.close()
        if env is not None: env.close()
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['rollouts']))


if __name__ == '__main__':
    main()
