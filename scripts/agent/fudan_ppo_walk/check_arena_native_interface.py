"""Compare the arena controller against the existing GPU action and observation path.

Only robot interface parity is checked; this is not field traversal acceptance.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sim2sim.rmuc2027_arena import (ArenaPlayback, build_model, CHECKPOINT,
    build_regional_model, regional_metadata)
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record
from ct_mjlab.fudan_ppo_walk.policy import load_policy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--terrain', choices=('plane', 'regions'), default='plane')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    policy, saved = load_policy(args.checkpoint)
    if args.terrain == 'regions':
        saved = regional_metadata(saved)
    model = build_regional_model(saved) if args.terrain == 'regions' else build_model(saved, arena=None)
    native = ArenaPlayback(model, saved)
    native.reset((0., 0., 0.))
    cfg = FudanWalkCfg()
    cfg.control = SimpleNamespace(**saved['control'])
    cfg.sim = SimpleNamespace(**{k: v for k, v in vars(cfg.sim).items() if not k.startswith('_')})
    cfg.sim.dt = saved['control_timing']['physics_dt']
    gpu = FudanEnv(1, randomized=False, cfg=cfg, reset_contract=saved['reset_contract'],
                   disturbances=from_record(saved['disturbances']))
    errors = dict(observation=0., torque=0., spring=0., policy=0.)
    try:
        action = gpu.action
        # 相同延迟、位置、速度和采样值；只比较控制实现，避免引擎轨迹分叉混入误差。
        native.delays.action_steps[:] = action.disturbances.action_steps.cpu()
        native.delays.sensor_steps[:] = action.disturbances.sensor_steps.cpu()
        native.delays.actions.zero_(); action.disturbances.actions.zero_()
        native.delays.action_head = action.disturbances.action_head = 0
        native.delays.sensors.copy_(action.disturbances.sensors.cpu())
        native.delays.sensor_head = action.disturbances.sensor_head
        native.pd_tick = action._pd_tick = 0
        rng = np.random.default_rng(20261115)
        with torch.inference_mode():
            for i in range(100):
                q = native.data.qpos.copy()
                q[native.q] += rng.uniform(-.1, .1, 6)
                rotation = rng.normal(size=4)
                q[3:7] = rotation / np.linalg.norm(rotation)
                v = rng.uniform(-2., 2., native.model.nv)
                native.data.qpos[:] = q; native.data.qvel[:] = v
                native.velocity[:] = rng.uniform(-3., 3., 6)
                native.last_action[:] = rng.uniform(-1., 1., 6)
                native.command[:] = rng.uniform([-.5, -.5, .22], [.5, .5, .30])
                gpu.env.sim.data.qpos[0] = torch.as_tensor(q, device=gpu.device)
                gpu.env.sim.data.qvel[0] = torch.as_tensor(v, device=gpu.device)
                action.sample_velocity[0] = torch.as_tensor(native.velocity, device=gpu.device)
                action._raw_actions[0] = torch.as_tensor(native.last_action, device=gpu.device)
                gpu.commands[0] = torch.as_tensor(native.command, device=gpu.device)
                gpu.env.sim.forward(); gpu.env.scene.update(dt=0.); gpu._update_state()
                sensed = native.sensors()
                native.delays.sample(torch.from_numpy(sensed[None]))
                action.disturbances.sample(action.sensor_frame())
                expected, _ = gpu.observations()
                actual = native.observation()
                errors['observation'] = max(errors['observation'], float(np.max(np.abs(actual - expected[0].cpu().numpy()))))
                history = rng.uniform(-1., 1., (1, 125)).astype(np.float32)
                a, _ = policy.act_inference(torch.from_numpy(actual[None]), torch.from_numpy(history))
                b, _ = policy.act_inference(expected.cpu(), torch.from_numpy(history))
                errors['policy'] = max(errors['policy'], float((a - b).abs().max()))
                requested = rng.uniform(-150., 150., 6).astype(np.float32)
                native.apply_controls(requested.clip(-native.normalization.clip, native.normalization.clip))
                action.process_actions(torch.as_tensor(requested[None], device=gpu.device))
                action.apply_actions()
                errors['torque'] = max(errors['torque'], float(np.max(np.abs(native.torque - action.torques[0].cpu().numpy()))))
                spring = gpu.env.sim.data.ctrl[0, action._spring_ids].cpu().numpy()
                errors['spring'] = max(errors['spring'], float(np.max(np.abs(native.data.ctrl[native.spring_ids] - spring))))
                assert native.pd_tick == action._pd_tick
        passed = max(errors.values()) < 3e-5
        report = dict(passed=passed, comparisons=100, errors=errors, timing=saved['control_timing'],
            checkpoint=str(args.checkpoint.resolve()), native_terrain=args.terrain,
            gpu_action_delay_steps=action.disturbances.action_steps.cpu().tolist(),
            gpu_sensor_delay_steps=action.disturbances.sensor_steps.cpu().tolist(),
            physics_steps=0, optimizer_updates=0,
            scope='Same-state observation/action/PD-hold/delay/spring parity against actual GPU implementation; not trajectory or field acceptance')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        assert passed, errors
    finally:
        gpu.close()


if __name__ == '__main__':
    main()
