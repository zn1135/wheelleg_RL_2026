"""Same-state native/GPU parity across actual RMUC model switches, without training."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import jump_config
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import from_record
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from sim2sim.arena_policies import ArenaPolicies, FieldContacts
from sim2sim.rmuc2027_arena import ArenaPlayback, REGIONAL_CHECKPOINT, build_model


def sync_state(gpu, native):
    sim = gpu.env.sim
    sim.data.qpos[0] = torch.as_tensor(native.data.qpos, device=gpu.device)
    sim.data.qvel[0] = torch.as_tensor(native.data.qvel, device=gpu.device)
    gpu.action.sample_velocity[0] = torch.as_tensor(native.velocity, device=gpu.device)


@torch.inference_mode()
def check():
    policies = ArenaPolicies(REGIONAL_CHECKPOINT)
    meta = policies.metadata['walk']
    env = ArenaPlayback(build_model(meta, arena=None), meta)
    env.reset((0., 0., 0.))
    policies.reset(env)
    refs = {role: deepcopy(model).cuda() for role, model in policies.policies.items()}
    errors = dict(frame=0., history=0., policy=0., torque=0., action_queue=0.)
    def compare(key, actual, reference):
        delta = np.abs(np.asarray(actual) - reference.cpu().numpy())
        assert np.isfinite(delta).all(), key
        errors[key] = max(errors[key], float(delta.max()))

    with ExitStack() as stack:
        cfg = FudanWalkCfg()
        cfg.control = SimpleNamespace(**meta['control'])
        cfg.sim.dt = meta['control_timing']['physics_dt']
        walk = FudanEnv(1, randomized=False, cfg=cfg, disturbances=from_record(meta['disturbances']))
        stack.callback(walk.close)
        jump = JumpEnv(1, randomized=False, cfg=jump_config(preload=True))
        stack.callback(jump.close)
        fifo = walk.action.disturbances
        for name in ('action_steps', 'sensor_steps', 'actions', 'sensors'):
            getattr(fifo, name).copy_(getattr(env.delays, name))
        fifo.action_head, fifo.sensor_head = env.delays.action_head, env.delays.sensor_head
        for gpu in (walk, jump):
            gpu.action._pd_tick = env.pd_tick
        contacts = FieldContacts(env.model)
        reference_history = {}
        roles = []
        base_control = env.apply_controls

        def frames():
            result = {}
            for gpu in (walk, jump):
                sync_state(gpu, env)
                gpu.commands[0] = torch.as_tensor(env.command, device=gpu.device)
                gpu.action._raw_actions[0] = torch.as_tensor(env.last_action, device=gpu.device)
                gpu.env.sim.forward()
                gpu.env.scene.update(dt=0.)
                gpu._update_state()
                result[gpu] = gpu.observations()[0]
            return result

        def paired_control(action, *, direct=False):
            # 两端都推进所有队列，逐毫秒比较实际生效控制器的力矩。
            base_control(action, direct=direct)
            for gpu in (walk, jump):
                sync_state(gpu, env)
                gpu.action.process_actions(torch.as_tensor(action[None], device=gpu.device))
                gpu.action.apply_actions()
            chosen = jump if direct else walk
            compare('torque', env.torque, chosen.action.torques[0])
            assert env.pd_tick == chosen.action._pd_tick

        env.apply_controls = paired_control
        samples = []
        def sample():
            sync_state(walk, env)
            fifo.sample(walk.action.sensor_frame())
            samples.append(contacts.sample(env.model, env.data))

        for tick in range(195):
            if tick in (10, 35):
                policies.select('step' if tick == 10 else 'walk', env)
            if tick == 50:
                assert policies.start_jump(env)
            env.command[0] = 2. if 10 <= tick < 24 else .3 if 35 <= tick < 50 else 0.
            obs, history = policies.inputs(env)
            current = frames()
            for gpu, frame in current.items():
                if tick == 0:
                    reference_history[gpu] = frame.repeat(1, 5)
                # 复位填满五帧后，第一条指令只能改最后一帧。
                reference_history[gpu][:, -19:-16] = frame[:, 6:9]
            chosen = jump if policies.gate is not None else walk
            compare('frame', obs, current[chosen][0])
            compare('history', history, reference_history[chosen][0])
            action, _ = policies.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(history[None]))
            role = policies.active_role
            expected, _ = refs[role].act_inference(current[chosen], reference_history[chosen])
            compare('policy', action, expected)
            roles.append(role)
            samples.clear()
            env.step(action[0].numpy(), direct=chosen is jump, sample=sample)
            policies.after_step(env, samples)
            compare('action_queue', env.delays.actions.numpy(), fifo.actions)
            for gpu, frame in frames().items():
                reference_history[gpu] = torch.cat((reference_history[gpu][:, 25:], frame), dim=1)
    assert set(roles) == set(refs) and roles[-1] == 'walk'
    transitions = [(i, role) for i, role in enumerate(roles) if not i or role != roles[i-1]]
    assert transitions[:5] == [(0, 'walk'), (10, 'step'), (24, 'parking'), (35, 'walk'), (50, 'jump')]
    assert len(transitions) == 6 and policies.transitions[-1]['reason'] == 'landed'
    # 持续轮地支撑而没有腾空时，验证另一条超时交接路径。
    from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import LANDING
    assert policies.start_jump(env)
    for _ in range(LANDING['max_jump_ticks']):
        assert policies.gate is not None
        env.steps += 1
        policies.after_step(env, [(65., 65., 0., 0.)] * 10)
    assert policies.gate is None and policies.transitions[-1]['reason'] == 'timeout'
    return dict(passed=max(errors.values()) < 3e-5, tolerance=3e-5, max_errors=errors,
                transitions=transitions, jump_events=policies.transitions, policy_steps=len(roles),
                native_physics_substeps=10*len(roles), synthetic_timeout_checked=True, optimizer_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = check()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)
    assert report['passed'], report
