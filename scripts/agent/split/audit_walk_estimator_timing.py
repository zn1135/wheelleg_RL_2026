#!/usr/bin/env python3
"""CPU-only storage/history audit; no simulation, training or optimizer steps."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from tensordict import TensorDict
from mjlab.utils.buffers.circular_buffer import CircularBuffer
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from ct_mjlab.envs.chuanliantui_split.network import HistoryActor, HistoryPPO
from ct_mjlab.envs.chuanliantui_split.env_cfg import make_ppo_cfg
from ct_mjlab.envs.chuanliantui.v14_observations import GroupedSensorDelay


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    torch.set_num_threads(1)
    torch.manual_seed(43)
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    actor_cfg = make_ppo_cfg('walk')
    assert not actor_cfg.actor.obs_normalization
    assert not actor_cfg.critic.obs_normalization
    obs_groups = {'actor': ['actor'], 'critic': ['critic']}

    def frame(t):
        return TensorDict(dict(actor=torch.full((2, 25), float(t)),
            history=torch.full((2, 125), float(t)),
            critic=torch.full((2, 71), float(t)),
            velocity_target=torch.full((2, 3), float(100 + t))), batch_size=[2])

    obs = frame(0)
    actor = HistoryActor(obs, obs_groups, 'actor', 6, hidden_dims=(8,),
        obs_normalization=False, distribution_cfg=dict(
            class_name='GaussianDistribution', init_std=.2, std_type='scalar'))
    critic = MLPModel(obs, obs_groups, 'critic', 1, hidden_dims=(8,), obs_normalization=False)
    storage = RolloutStorage('rl', 2, 3, obs, (6,))
    algorithm = HistoryPPO(actor, critic, storage, device='cpu')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No learning allowed')):
        with torch.inference_mode():
            for tick in range(3):
                current = frame(tick)
                algorithm.act(current)
                # Next obs deliberately carries different labels, exposing an off-by-one copy.
                algorithm.process_env_step(frame(tick + 1), torch.zeros(2),
                    torch.zeros(2, dtype=torch.bool), {})
    stored = storage.observations
    for tick in range(3):
        assert (stored['history'][tick] == tick).all()
        assert (stored['velocity_target'][tick] == 100 + tick).all()
    retained = stored.clone()
    storage.clear()
    assert storage.step == 0
    assert all(torch.equal(stored[key], retained[key]) for key in retained.keys())

    history = CircularBuffer(5, 2, 'cpu')
    for tick in range(7):
        history.append(torch.full((2, 25), float(tick)))
    assert history.buffer[0, :, 0].tolist() == [2., 3., 4., 5., 6.]
    assert (history.buffer.reshape(2, -1)[:, -25:] == 6).all()
    delay = GroupedSensorDelay(torch.zeros(2, 18), 8)
    delay.lags[:] = torch.tensor([0, 3, 5, 8])
    for tick in range(10):
        delay.push(torch.full((2, 18), float(tick)))
    assert delay.read()[0].tolist() == [9.]*3 + [6.]*3 + [4.]*6 + [1.]*6

    relative = [
        'envs/chuanliantui_split/network.py', 'envs/chuanliantui_split/observations.py',
        'envs/chuanliantui_split/env_cfg.py', 'envs/chuanliantui_stand_v2/network.py',
        'envs/chuanliantui_stand_v2/observations.py', 'envs/chuanliantui_stand_v2/env_cfg.py',
        'envs/chuanliantui/v14_observations.py', 'envs/chuanliantui/huananhu_env_cfg.py',
        'envs/chuanliantui/multimode_env_cfg.py']
    current = root / 'mjlab_training/src/ct_mjlab'
    frozen = args.frozen / 'mjlab_training/src/ct_mjlab'
    comparison = [dict(path=name, current_sha256=digest(current/name),
        frozen_sha256=digest(frozen/name), equal=(current/name).read_bytes()==(frozen/name).read_bytes())
        for name in relative]
    assert all(row['equal'] for row in comparison)
    result = dict(verified=True, simulation_steps=0, optimizer_steps=0, device='cpu',
        scope='Synthetic data-flow and static source audit only; not a behavior acceptance test',
        storage_same_frame_history_and_target=True, buffer_clear_keeps_tensors=True,
        history_oldest_to_newest=True, current_actor_frame_is_history_last25=True,
        grouped_sensor_lag_direction_correct=True,
        actor_observation_normalization=False, critic_observation_normalization=False,
        history_observations=5, policy_dt_ms=10, history_oldest_age_ms=40,
        configured_sensor_delay_ms=[5,10,15], actual_nearest_2ms_delays_ms=[6,10,16],
        train_play_condition_difference='Training has noise and grouped sensor delays; nominal play disables both. Motor/physics randomization and stochastic actions also differ.',
        velocity_scaling='Supervision COM velocity *2; actor consumes encoder directly; export reports encoder/2. Oracle substitutes same COM*2.',
        no_identified_off_by_one_or_scaling_bug=True, source_comparison=comparison,
        caveat='External mjlab/rsl_rl packages were inspected as currently installed; they were not frozen with the training source snapshot.')
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='source_comparison'},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
