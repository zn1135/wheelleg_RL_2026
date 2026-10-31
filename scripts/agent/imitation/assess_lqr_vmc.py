#!/usr/bin/env python3
"""Finite teacher behavior assessment; no training, policy or automatic resets."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from ct_mjlab.teacher.env import make_teacher_env_cfg
from ct_mjlab.teacher.mechanics import DT
from ct_mjlab.scripts.teacher import Recorder, ROOT, run


def sequence_check(args):
    rows = json.loads((ROOT / 'mjlab_training/configs/teacher_motion.json').read_text())
    for i, row in enumerate(rows):
        row['time'] = i * .01
    with tempfile.TemporaryDirectory(prefix='teacher-sequence-') as tmp:
        path = Path(tmp) / 'sequence.json'
        path.write_text(json.dumps(rows))
        run(SimpleNamespace(num_envs=2, device=args.device, seed=args.seed, seconds=.07,
                            headless=True, viewer='auto', initial='standing', vx=0., yaw=0.,
                            leg_length=.15, sequence=path, output=args.output,
                            velocity_integral=args.velocity_integral))
    with np.load(args.output / 'trace.npz') as trace:
        expected = np.asarray([[r[k] for k in ('vx','yaw','leg_length')] for r in rows])
        np.testing.assert_allclose(trace['command'], np.repeat(expected[:,None],2,axis=1), atol=1e-7)
        assert trace['qpos'].shape[0:2] == (7, 2)
    result_path = args.output / 'result.json'
    result = json.loads(result_path.read_text())
    result['sequence_dispatch_checked'] = True
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('motion', 'ground', 'standing', 'reset', 'sequence'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--leg-length', type=float, help='Override the diagnostic grid leg-length command')
    parser.add_argument('--velocity-integral', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.case == 'sequence':
        sequence_check(args)
        return
    args.initial = 'ground' if args.case == 'ground' else 'standing'
    commands = [[0., 0., .15]]
    args.seconds = 20.
    if args.case == 'motion':
        args.seconds = 12.
        commands = [[0.,0.,.15], [.5,0.,.15], [-.5,0.,.15],
                    [0.,.5,.15], [0.,-.5,.15], [0.,0.,.22], [0.,0.,.30],
                    [.5,0.,.22], [-.5,0.,.22], [0.,.5,.22], [0.,-.5,.22]]
    elif args.case == 'ground':
        commands *= 20
    elif args.case == 'reset':
        args.seconds = .05
        commands *= 20
    if args.leg_length is not None:
        if not .13 <= args.leg_length <= .34:
            raise ValueError('leg-length must be within0.13~0.34m')
        commands = [[vx,yaw,args.leg_length] for vx,yaw,_ in commands]
        if args.case == 'motion':
            commands = [list(row) for row in dict.fromkeys(map(tuple,commands))]
    args.num_envs = len(commands)
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch_backends()
    cfg = make_teacher_env_cfg(initial=args.initial, num_envs=args.num_envs, seed=args.seed,
                               velocity_integral=args.velocity_integral)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    recorder, complete = None, False
    reset_checked = False
    inverted_checked = False
    try:
        env.reset(seed=args.seed)
        command = torch.tensor(commands, device=args.device)
        recorder = Recorder(env, args.output, args, [{'time': 0., 'commands_per_env': commands}])
        with torch.inference_mode():
            for tick in range(round(args.seconds/DT)):
                env.step(command)
                recorder.count += 1
                recorder.sample(command)
                if args.case == 'reset' and tick == 24:
                    term = env.action_manager.get_term('teacher')
                    untouched = torch.tensor([i for i in range(20) if i not in (2,7)], device=args.device)
                    before = env.sim.data.qpos[untouched].clone()
                    states = {k: v[untouched].clone() for k,v in vars(term.controller).items()
                              if isinstance(v, torch.Tensor)}
                    env.reset(env_ids=torch.tensor([2,7], device=args.device))
                    torch.testing.assert_close(env.sim.data.qpos[untouched], before, atol=0, rtol=0)
                    for k,v in states.items():
                        torch.testing.assert_close(getattr(term.controller,k)[untouched], v, atol=0, rtol=0)
                    assert not term.controller.pos[[2,7]].any()
                    assert not term.controller.velocity.ready[[2,7]].any()
                    assert not term.controller.length_pd.ready[[2,7]].any()
                    assert not term.controller.standup.elapsed[[2,7]].any()
                    reset_checked = True
                if args.case == 'reset' and tick == 34:
                    robot = env.scene['robot']
                    ids = torch.tensor([3], device=args.device)
                    root = robot.data.default_root_state[ids].clone()
                    root[:, :3] = env.scene.env_origins[ids]
                    root[:, 2] += 1.
                    root[:, 3:7] = root.new_tensor([0.,1.,0.,0.])
                    root[:, 7:] = 0
                    robot.write_root_state_to_sim(root, env_ids=ids)
                    env.sim.forward()
                if args.case == 'reset' and tick >= 36:
                    term = env.action_manager.get_term('teacher')
                    assert term.controller.standup.fault[3] == 2
                    assert not term.torques[3].any()
                    assert not env.sim.data.actuator_force[3, term.motor_actuators].any()
                    inverted_checked = True
        complete = True
    except Exception as exc:
        if recorder is not None:
            recorder.run_error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if recorder is not None:
            recorder.save(complete)
            result_path = args.output / 'result.json'
            result = json.loads(result_path.read_text())
            result['commands_per_env'] = commands
            result['partial_reset_checked'] = reset_checked
            result['inverted_motor_zero_checked'] = inverted_checked
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        env.close()


if __name__ == '__main__':
    main()
