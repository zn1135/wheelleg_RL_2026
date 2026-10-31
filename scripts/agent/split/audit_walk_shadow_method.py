#!/usr/bin/env python3
"""CPU source/hash and sentinel audit for the frozen shadow-delay experiment."""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root = Path(__file__).resolve().parents[3]
    artifact = args.artifact.resolve()
    launch = json.loads((args.training_artifact/'launch.json').read_text())
    sealed = json.loads((args.training_artifact/'source-sha256.json').read_text())
    frozen = Path(launch['log_dir'])/'snapshot/ct_mjlab'
    files = (
        'envs/chuanliantui/v14_observations.py',
        'envs/chuanliantui/huananhu_observations.py',
        'envs/chuanliantui/multimode_observations.py',
        'envs/chuanliantui_stand_v2/observations.py',
        'envs/chuanliantui_stand_v2/env_cfg.py',
        'envs/chuanliantui_split/observations.py',
        'envs/chuanliantui_split/env_cfg.py',
        'envs/chuanliantui_split/network.py',
    )
    source_audit = []
    for name in files:
        current = root/'mjlab_training/src/ct_mjlab'/name
        saved = frozen/name
        row = dict(relative_path=name, production_path=str(current), frozen_path=str(saved),
            production_sha256=sha(current), frozen_sha256=sha(saved), training_manifest_sha256=sealed[name])
        row['identical'] = row['production_sha256'] == row['frozen_sha256'] == row['training_manifest_sha256']
        assert row['identical'], name
        source_audit.append(row)

    external = []
    for module in ('mjlab.managers.observation_manager', 'mjlab.utils.buffers.circular_buffer',
                   'mjlab.envs.manager_based_rl_env', 'mjlab.managers.metrics_manager'):
        path = Path(importlib.util.find_spec(module).origin)
        external.append(dict(module=module, import_path=str(path), resolved_path=str(path.resolve()), sha256=sha(path)))

    from ct_mjlab.envs.chuanliantui.v14_observations import GroupedSensorDelay, SENSOR_SLICES
    from ct_mjlab.envs.chuanliantui_stand_v2.observations import StandV2ObservationState
    from mjlab.utils.buffers.circular_buffer import CircularBuffer

    card = json.loads((artifact/'experiment-card.json').read_text())
    combinations = tuple(itertools.product((3, 5, 8), repeat=4))
    assert list(map(list, combinations)) == card['combination_order']
    assert len(combinations) == 81
    env_count, ticks = len(combinations), 20
    # 每个时间、环境、槽位都有不同值，检查分组索引而不引入任何真速度标签。
    primitive = torch.arange((5*ticks+1)*env_count*18, dtype=torch.float32).reshape(5*ticks+1, env_count, 18)*1e-7
    delay = GroupedSensorDelay(primitive[0], 8)
    delay.lags[:] = torch.tensor(combinations)
    history = CircularBuffer(5, env_count, 'cpu')
    fake = SimpleNamespace(delay=delay, cfg=SimpleNamespace(noise=False), env=SimpleNamespace(device='cpu'),
        command=SimpleNamespace(command=torch.zeros(env_count, 3)),
        action=SimpleNamespace(raw_action=torch.zeros(env_count, 6)),
        mode=lambda: torch.empty(env_count, 0))
    reference_frames = []
    residuals = dict(grouped_sensor=0., actor25=0., history125=0.)
    for tick in range(ticks):
        if tick:
            for index in range(5*(tick-1)+1, 5*tick+1):
                delay.push(primitive[index])
        expected = torch.cat([primitive[
            torch.clamp(5*tick-delay.lags[:, group], min=0), torch.arange(env_count), columns]
            for group, columns in enumerate(SENSOR_SLICES)], -1)
        residuals['grouped_sensor'] = max(residuals['grouped_sensor'], float((delay.read()-expected).abs().max()))
        fake.command.command.fill_(.01*tick)
        fake.action.raw_action.fill_(-.01*tick)
        frame = StandV2ObservationState.actor(fake)
        expected_frame = torch.cat((expected[:, :3]*.25, expected[:, 3:6],
            fake.command.command*torch.tensor((2., .25, 5.)), expected[:, 6:10],
            expected[:, 12:18][:, (0, 1, 4, 2, 3, 5)]*.05, fake.action.raw_action), -1).clamp(-100., 100.)
        assert frame.shape == (env_count, 25)
        residuals['actor25'] = max(residuals['actor25'], float((frame-expected_frame).abs().max()))
        reference_frames.append(expected_frame)
        expected_history = torch.cat([reference_frames[max(0, tick-lag)] for lag in (4, 3, 2, 1, 0)], -1)
        history.append(frame)
        residuals['history125'] = max(residuals['history125'], float((history.buffer.reshape(env_count, -1)-expected_history).abs().max()))
    assert all(value == 0. for value in residuals.values())
    collector = root/'scripts/agent/split/record_walk_shadow_sensors.py'
    lock = json.loads((artifact/'capture-lock.json').read_text())
    assert sha(collector) == lock['collector_sha256']
    assert sha(artifact/'experiment-card.json') == lock['card_sha256']
    result = dict(verified=True, scope='Source consistency and CPU sentinel only; not physics or behavior acceptance',
        source_files=source_audit, runtime_external_dependencies=external,
        external_history_limit='Original training artifact seals ct_mjlab and physical assets, not external mjlab dependencies. Current external implementation and hashes are recorded, but historical dependency identity cannot be proved.',
        collector_path=str(collector), collector_sha256=sha(collector),
        experiment_card_sha256=sha(artifact/'experiment-card.json'),
        synthetic=dict(combinations=81, policy_ticks=ticks, physics_samples=5*ticks+1,
            reset_fill='S0 for negative raw indices; first Actor25 for all five history slots',
            labels_used=False, residuals=residuals),
        timing=dict(raw_group_index='max(0, 5*k-lag_g)', raw_dt_s=.002, policy_dt_s=.01,
            group_order=['gyro 0:3','gravity 3:6','centered q 6:12','dq 12:18'],
            lag_samples=[3,5,8], realized_ms=[6,10,16],
            history='A[max(0,k-4)] ... A[k], old-to-new, append once per policy tick',
            command_and_previous_action='Native current-policy slots; never delayed',
            labels='Current pre-action root-body COM velocity in body frame; untouched; scored only',
            external_policy_forward='Current manager calls sim.forward after physics and before command/observation; no additional <=2ms label shift is applied',
            correlation_limit='Output correlation lag is descriptive, not a fixed physical sensor or estimator delay'),
        production_changes=False, training_updates=0, shadow_used_for_control=False)
    (artifact/'method-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(verified=True, source_files=len(source_audit), external_dependencies=len(external),
        synthetic=result['synthetic']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    main(parser.parse_args())
