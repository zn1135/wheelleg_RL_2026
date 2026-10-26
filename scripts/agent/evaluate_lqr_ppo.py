#!/usr/bin/env python3
"""Paired cached-pose nominal20s benchmark: native1ms LQR, BC or real PPO actor."""
import argparse, hashlib, json, math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.os import dump_yaml

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.ppo import configure_smp, checkpoint_bc_finetune, checkpoint_smp_weight, policy_observation_contract
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balance_probe(env):
    p = getattr(env, '_comparison_probe', None)
    if p is None: return torch.zeros(env.num_envs, device=env.device)
    data = env.scene['robot'].data; g = data.projected_gravity_b
    roll = torch.atan2(-g[:, 1], -g[:, 2]); pitch = torch.asin(g[:, 0].clamp(-1, 1))
    wheels = -env.scene['wheel_contact'].data.force[:, :, 2]
    base = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
    angular = data.root_link_ang_vel_b[:, :2].norm(dim=-1)
    height = data.root_link_pos_w[:, 2]-env.scene.env_origins[:, 2]
    leg = env.sim.data.qvel[:, p['leg_vadr']]
    fields = torch.stack((wheels[:, 0], wheels[:, 1], base, roll, pitch, angular, height,
                          p['live'].float(), *leg.unbind(-1)), -1)
    p['physical'].append(fields)
    tilt = torch.maximum(roll.abs(), pitch.abs()); weight = p['weight']
    balanced = ((wheels >= max(.1*weight, 5.)).all(-1) & (wheels.sum(-1) >= .8*weight)
                & (wheels.sum(-1) <= 1.2*weight) & (base <= 1.)
                & (tilt <= math.radians(35)) & (angular <= .5))
    strict = balanced & (tilt <= math.radians(15)) & ((height-.33648).abs() <= .03)
    p['hold'] = torch.where(p['live'], torch.where(balanced, p['hold']+1, 0), 0)
    p['strict_run'] = torch.where(p['live'], torch.where(strict, p['strict_run']+1, 0), 0)
    p['strict_ever'] |= p['live'] & (p['strict_run'] >= round(.5/env.physics_dt))
    return balanced.float()


@torch.inference_mode()
def run(args):
    from mjlab.utils.torch import configure_torch_backends
    torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=False)
    source_at_start = digest(__file__)
    cached_path = args.source/'nominal_mean/initial-state.pt'
    cached = torch.load(cached_path, map_location='cpu', weights_only=False)
    baseline = json.loads((args.source/'nominal_mean/result.json').read_text())
    indices = list(range(10 if args.kind == 'native' else 40))
    if args.kind == 'native':
        from ct_mjlab.teacher.env import make_teacher_env_cfg
        from mjlab.envs import ManagerBasedRlEnv
        cfg = make_teacher_env_cfg(initial='ground', num_envs=10, seed=42, velocity_integral=True)
        cfg.metrics['comparison'] = MetricsTermCfg(func=balance_probe, per_substep=True, reduce='last')
        env = ManagerBasedRlEnv(cfg, device='cuda:0'); raw = env
        train = None
    else:
        cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
        contract = policy_observation_contract(args.checkpoint, bc=args.kind == 'bc')
        assert contract == HANDOVER_OBSERVATION_CONTRACT
        configure_smp(cfg, train, args.prior, play=True, policy_contract=contract,
                      reward_weight=checkpoint_smp_weight(args.checkpoint) if args.kind == 'ppo' else 1.,
                      bc_finetune=checkpoint_bc_finetune(args.checkpoint) if args.kind == 'ppo' else True)
        cfg.seed = train.seed = 42
        cfg.actions['pd'].randomize_delays = False
        cfg.metrics['comparison'] = MetricsTermCfg(func=balance_probe, per_substep=True, reduce='last')
        env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=len(indices))
        raw = env.unwrapped
    configure_torch_backends(allow_tf32=False)
    try:
        cfg.auto_reset = False; raw.cfg.auto_reset = False
        if args.kind == 'ppo':
            runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
            runner.load(str(args.checkpoint), load_cfg={'actor': True})
            actor = runner.alg._raw_actor
        elif args.kind == 'bc':
            from ct_mjlab.bc import load_actor
            actor, _ = load_actor(args.checkpoint, raw.device)
        raw.reset(seed=42)
        q = cached['qpos'][indices].to(raw.device).clone(); q[:, :2] = raw.scene.env_origins[:, :2]
        v = cached['qvel'][indices].to(raw.device).clone()
        raw.sim.data.qpos.copy_(q); raw.sim.data.qvel.copy_(v); raw.sim.forward()
        if args.kind == 'native':
            action = raw.action_manager.get_term('teacher'); action.reset()
            output = torch.tensor([0., 0., .30], device=raw.device).expand(len(indices), -1)
            obs = None
        else:
            action = raw.action_manager.get_term('pd'); action.reset()
            action.action_delay_steps.copy_(cached['leg_delay'].to(raw.device))
            action.wheel_action_delay_steps.copy_(cached['wheel_delay'].to(raw.device))
            command = raw.command_manager.get_term('standup'); command.ground_start[:] = True; command.compute(dt=0.)
            multimode_observations.reset_observations(raw, torch.arange(len(indices), device=raw.device))
            obs = raw.observation_manager.compute(update_history=True)
        weight = float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        p = dict(weight=weight, live=torch.ones(len(indices), dtype=torch.bool, device=raw.device),
            hold=torch.zeros(len(indices), dtype=torch.long, device=raw.device),
            strict_run=torch.zeros(len(indices), dtype=torch.long, device=raw.device),
            strict_ever=torch.zeros(len(indices), dtype=torch.bool, device=raw.device), physical=[])
        raw._comparison_probe = p; reader = MotionStateReader(raw)
        p['leg_vadr'] = reader.vadr[[0, 1, 3, 4]]
        buffers = {key: [] for key in ('qpos', 'qvel', 'motor_dq', 'features', 'action', 'alive', 'terminated', 'time_out')}
        stride = round(.01/raw.step_dt); steps = round(20./raw.step_dt)
        for tick in range(steps):
            live = p['live'].clone()
            if args.kind != 'native': output = actor(obs)
            result = env.step(output)
            if args.kind == 'native':
                from ct_mjlab.teacher.standup import FAILED
                done = action.controller.standup.phase == FAILED
                time_out = torch.zeros_like(done)
            else:
                obs, _, _, _ = result
                # 完整20秒到期属于正常观察结束，不能当作物理失效。
                done = raw.termination_manager.terminated
                time_out = raw.termination_manager.time_outs
                if tick < 20: assert not action.torques.any()
            if (tick+1) % stride == 0:
                state = reader.read()
                values = dict(qpos=raw.sim.data.qpos, qvel=raw.sim.data.qvel, motor_dq=state['motor_dq'],
                    features=reader.features(), action=output, alive=live, terminated=done, time_out=time_out)
                for key, value in values.items(): buffers[key].append(value.cpu().numpy().copy())
            p['live'] &= ~done.bool()
            if (tick+1) % round(5./raw.step_dt) == 0:
                print(f'{args.kind} t={(tick+1)*raw.step_dt:.1f}s alive={int(p["live"].sum())}/{len(indices)}', flush=True)
        arrays = {key: np.asarray(value) for key, value in buffers.items()}
        physical_tensor = torch.stack(p['physical'])
        rows = []
        for i in indices:
            n = int(arrays['alive'][:, i].sum()); full = n == 2000
            finite = all(np.isfinite(arrays[key][:n, i]).all() for key in ('qpos', 'qvel', 'motor_dq', 'features'))
            late = slice(1500, n)
            rms = float(np.sqrt(np.square(arrays['motor_dq'][late, i][:, [0, 1, 3, 4]]).mean())) if n > 1500 else None
            speed = float(np.linalg.norm(arrays['features'][late, i, 4:6], axis=-1).mean()) if n > 1500 else None
            drift = float(np.linalg.norm(arrays['qpos'][late, i, :2]-arrays['qpos'][1500, i, :2], axis=-1).max()) if n > 1500 else None
            yaw = float(np.abs(arrays['features'][late, i, 9]).mean()) if n > 1500 else None
            balanced = bool(full and p['live'][i] and p['hold'][i] >= round(1./raw.physics_dt))
            rise = bool(p['strict_ever'][i])
            physical_leg_rms = float(physical_tensor[round(15./raw.physics_dt):, i, 8:12].square().mean().sqrt()) if full else None
            rows.append(dict(env=i, group_id=baseline['rows'][i].get('group_id', str(i % 10)),
                leg_delay_s=float(cached['leg_delay'][i])*.002 if args.kind != 'native' else None,
                wheel_delay_s=float(cached['wheel_delay'][i])*.002 if args.kind != 'native' else None,
                strict_rise=rise, final_balance_v31=balanced, full_20s=full, finite=finite,
                terminated=not bool(p['live'][i]),
                last5s_leg_velocity_rms_rad_s=rms, last5s_mean_speed_m_s=speed,
                last5s_xy_excursion_m=drift, last5s_mean_abs_yaw_rate_rad_s=yaw,
                last5s_physical_leg_velocity_rms_rad_s=physical_leg_rms,
                qualified=bool(rise and balanced and finite and rms is not None and rms <= .02
                               and speed <= .01 and drift <= .02 and yaw <= .02)))
        physical = physical_tensor.cpu().numpy()
        np.savez(args.output/'trace.npz', **arrays, physical=physical, initial_qpos=q.cpu().numpy(),
                 initial_qvel=v.cpu().numpy(), time=(np.arange(2000)+1)*.01)
        dump_yaml(args.output/'env.yaml', asdict(cfg))
        report = dict(completed=True, kind=args.kind, optimizer_updates=0, rows=rows,
            strict_rise=sum(x['strict_rise'] for x in rows), final_balance=sum(x['final_balance_v31'] for x in rows),
            qualified=sum(x['qualified'] for x in rows), num_envs=len(indices), physical_dt=raw.physics_dt,
            command=[0., 0., .30 if args.kind == 'native' else .33648],
            height_command_kind='leg_length' if args.kind == 'native' else 'base_height',
            ground_hold_s=0. if args.kind == 'native' else .2,
            weight_n=weight, xml_sha256=digest(MODEL_XML), cached_initial_sha256=digest(cached_path),
            checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
            source_sha256=source_at_start, source_sha256_end=digest(__file__), trace_sha256=digest(args.output/'trace.npz'),
            scope='Native1ms LQR at its original0s hold vs student2ms PD/100Hz with0.2s hold. Same cached10 poses; student repeats four2/4ms delay pairs. Report control differences, no hardware/generalization claim.')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
    finally: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('native', 'ppo', 'bc'), required=True)
    for name in ('source', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    for name in ('checkpoint', 'prior'): parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.kind != 'native' and (args.checkpoint is None or args.prior is None): parser.error('Student requires checkpoint and prior')
    run(args)
