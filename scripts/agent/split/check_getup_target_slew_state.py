#!/usr/bin/env python3
"""CPU checks for TargetSlew state only; no simulation or training."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.agent.split.assess_getup_target_slew import TargetSlew


LEGS = torch.tensor([0, 1, 3, 4])
WHEELS = torch.tensor([2, 5])


def near(actual, expected, label, atol=2e-6):
    actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
    error = float((actual-expected).abs().max())
    assert error <= atol, f'{label}: {error} > {atol}'
    return error


class FakePD:
    """Only the original FIFO, target and PD arithmetic needed by the wrapper."""
    def __init__(self, raw):
        self.raw = raw
        self.cfg = SimpleNamespace(leg_position_scale=.5)
        self._leg_action_ids, self._wheel_action_ids = LEGS, WHEELS
        self._motor_q_adr = self._motor_v_adr = torch.arange(6)
        self._env_ids = torch.arange(raw.num_envs)
        self._history_head = 1
        self._action_history = torch.zeros(4, raw.num_envs, 6)
        self._control_actions = torch.zeros(raw.num_envs, 6)
        self.raw_action = torch.zeros_like(self._control_actions)
        self._filtered_leg_actions = torch.zeros(raw.num_envs, 4)
        self._nominal_pos = torch.zeros_like(self._control_actions)
        self.reference_positions = torch.zeros_like(self._control_actions)
        self.action_delay_steps = torch.zeros(raw.num_envs, dtype=torch.long)
        self.wheel_action_delay_steps = torch.zeros_like(self.action_delay_steps)
        self.torques = torch.zeros_like(self._control_actions)
        self.apply_count, self.reset_count = 0, 0
        self.records = []
        self.raise_on_apply = False

    @property
    def nominal_positions(self):
        return self._nominal_pos

    def _ground_hold(self):
        return self.raw.state.ground_start & (self.raw.episode_length_buf < 20)

    def process(self, action):
        # Only the policy call advances EMA; TargetSlew.apply must leave it alone.
        self.raw_action.copy_(action)
        self._control_actions.copy_(action)
        self._filtered_leg_actions.mul_(.2).add_(action[:, LEGS], alpha=.8)
        self._filtered_leg_actions[self._ground_hold()] = 0.
        self._control_actions[:, LEGS] = self._filtered_leg_actions

    def apply_actions(self):
        self.apply_count += 1
        self._history_head = (self._history_head+1) % len(self._action_history)
        self._action_history[self._history_head].copy_(self._control_actions)
        legs = self._action_history[(self._history_head-self.action_delay_steps)
            % len(self._action_history), self._env_ids]
        wheels = self._action_history[(self._history_head-self.wheel_action_delay_steps)
            % len(self._action_history), self._env_ids]
        delayed = legs.clone()
        delayed[:, WHEELS] = wheels[:, WHEELS]
        target = (self._nominal_pos[:, LEGS]+.5*delayed[:, LEGS]).clamp(
            self.reference_positions[:, LEGS]-3.14, self.reference_positions[:, LEGS]+3.14)
        self.torques[:, LEGS] = 60.*(target-self.raw.sim.data.qpos[:, LEGS])-2.*self.raw.sim.data.qvel[:, LEGS]
        self.torques[:, WHEELS] = delayed[:, WHEELS]-self.raw.sim.data.qvel[:, WHEELS]
        self.torques.clamp_(-torch.tensor([40.,40.,3.9,40.,40.,3.9]),
            torch.tensor([40.,40.,3.9,40.,40.,3.9]))
        self.torques[self._ground_hold()] = 0.
        self.records.append(dict(target=target.clone(), delayed=delayed.clone(),
            nominal=self._nominal_pos.clone(), torque=self.torques.clone(),
            head=self._history_head, history=self._action_history.clone()))
        if self.raise_on_apply:
            raise RuntimeError('synthetic original apply failure')

    def reset(self, env_ids=None):
        self.reset_count += 1
        ids = slice(None) if env_ids is None else env_ids
        self.raw_action[ids] = 0.
        self._control_actions[ids] = 0.
        self._filtered_leg_actions[ids] = 0.
        self.torques[ids] = 0.
        self._action_history[:, ids] = 0.


def fixture(rate=2.):
    state = SimpleNamespace(ground_start=torch.tensor([True, True, False]),
        has_stood=torch.zeros(3, dtype=torch.bool))
    raw = SimpleNamespace(num_envs=3, device='cpu', physics_dt=.002,
        episode_length_buf=torch.zeros(3, dtype=torch.long), state=state,
        command_manager=SimpleNamespace(get_term=lambda name: state),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(3, 6),
            qvel=torch.zeros(3, 6))))
    pd = FakePD(raw)
    return raw, pd, TargetSlew(pd, raw, rate)


def apply_checked(slew):
    """Independent next-head oracle also detects a second FIFO push."""
    pd = slew.pd
    preserved = {name: getattr(pd, name).clone() for name in
        ('raw_action', '_control_actions', '_filtered_leg_actions', '_nominal_pos')}
    expected_history = pd._action_history.clone()
    head = (pd._history_head+1) % len(expected_history)
    expected_history[head] = preserved['_control_actions']
    delayed = expected_history[(head-pd.action_delay_steps) % len(expected_history), pd._env_ids].clone()
    wheel_delayed = expected_history[(head-pd.wheel_action_delay_steps) % len(expected_history), pd._env_ids]
    delayed[:, WHEELS] = wheel_delayed[:, WHEELS]
    expected_desired = (slew.original_nominal[:, LEGS]+.5*delayed[:, LEGS]).clamp(
        pd.reference_positions[:, LEGS]-3.14, pd.reference_positions[:, LEGS]+3.14)
    calls = pd.apply_count
    slew.apply()
    assert pd.apply_count == calls+1 and pd._history_head == head
    assert torch.equal(pd._action_history, expected_history), 'exactly one original FIFO write'
    for name, expected in preserved.items():
        assert torch.equal(getattr(pd, name), expected), f'wrapper changed {name}'
    record = pd.records[-1]
    assert torch.equal(record['delayed'], delayed), 'original apply reads same delayed actions'
    near(slew.desired, expected_desired, 'next-head desired')
    near(record['target'], slew.applied, 'original apply sees limited target')
    # Channels, damping and limits in the fake original remain visible.
    expected_wheel = (delayed[:, WHEELS]-slew.raw.sim.data.qvel[:, WHEELS]).clamp(-3.9,3.9)
    expected_wheel[pd._ground_hold()] = 0.
    near(record['torque'][:, WHEELS], expected_wheel, 'wheel path unchanged')
    return record


def check_hold_release():
    raw, pd, slew = fixture()
    raw.sim.data.qvel[:, LEGS] = torch.tensor([.2,-.3,.4,-.5])
    action = torch.full((3,6), .4)
    for tick in range(20):
        raw.episode_length_buf.fill_(tick)
        pd.process(action)
        for substep in range(5):
            raw.sim.data.qpos[:, LEGS] = .001*(tick*5+substep)
            apply_checked(slew)
            assert not slew.started[:2].any(), 'no constructor/hold latch'
            assert not pd.torques[:2].any(), 'all six held motors zero'
    q_release = torch.tensor([[.21,-.32,.43,-.54],[.11,-.22,.33,-.44]])
    raw.sim.data.qpos[:2, LEGS] = q_release
    raw.episode_length_buf.fill_(20)
    pd.process(action)
    record = apply_checked(slew)
    assert slew.started.tolist() == [True,True,False]
    near(record['target'][:2], q_release, 'first active measured-angle latch')
    near(record['torque'][:2, LEGS], -2.*raw.sim.data.qvel[:2, LEGS],
        'first active damping retained', atol=1e-5)
    near(record['target'][2], slew.desired[2], 'non-ground bypass')
    return dict(wait_substeps=100, first_active_substep=100,
        measured_angle_latched_at_release=True, first_active_damping_preserved=True,
        no_latch_for_non_ground=True)


def check_fifo():
    raw, pd, slew = fixture(rate=0.)
    raw.episode_length_buf.fill_(20)
    pd.action_delay_steps[:] = torch.tensor([0,2,1])
    pd.wheel_action_delay_steps[:] = torch.tensor([2,0,2])
    pd._action_history[:] = torch.arange(72).reshape(4,3,6)*.001
    seen = []
    for step in range(12):
        pd.process(torch.arange(18).reshape(3,6)*.03+step*.02)
        record = apply_checked(slew)
        seen.append(record['delayed'])
        near(record['delayed'][0, LEGS], pd._control_actions[0, LEGS],
            'delay zero reads new head instead of overwritten old slot')
        near(record['delayed'][1, WHEELS], pd._control_actions[1, WHEELS],
            'wheel zero delay independent of leg delay')
    assert not slew.started.any() and not slew.finished.any(), 'rate zero bypass'
    # A constant command after a reset arrives exactly after two original pushes.
    slew.reset()
    pd.action_delay_steps.fill_(2)
    pd.wheel_action_delay_steps.fill_(2)
    pd._control_actions.fill_(.3)
    release = [apply_checked(slew)['delayed'] for _ in range(3)]
    assert not release[0].any() and not release[1].any()
    near(release[2], torch.full((3,6),.3), 'fixed delay two physics substeps')
    return dict(mixed_leg_delay_steps=[0,2,1], mixed_wheel_delay_steps=[2,0,2],
        ring_rollovers_checked=3, fixed_two_substep_delay=True,
        zero_delay_reads_new_slot=True, one_fifo_write_per_apply=True,
        raw_control_ema_and_nominal_preserved=True, rate_zero_bypass=True)


def check_rate_reverse_handoff():
    raw, pd, slew = fixture()
    raw.state.ground_start.fill_(True)
    raw.episode_length_buf.fill_(20)
    pd._control_actions[:, LEGS] = .8
    apply_checked(slew)
    targets = [slew.applied.clone()]
    for _ in range(10):
        apply_checked(slew)
        targets.append(slew.applied.clone())
    before_reverse = slew.applied.clone()
    pd._control_actions[:, LEGS] = -.4
    for _ in range(10):
        apply_checked(slew)
        targets.append(slew.applied.clone())
    assert (slew.applied < before_reverse).all(), 'target reversal follows negative direction'
    raw.state.has_stood.fill_(True)
    assert not slew.finished.any(), 'standing alone must not bypass while chasing'
    finish_step = None
    for step in range(100):
        previous = slew.applied.clone()
        record = apply_checked(slew)
        targets.append(slew.applied.clone())
        if slew.finished.all():
            finish_step = step
            near(slew.applied, slew.desired, 'catch-up target before handoff')
            assert float((slew.applied-previous).abs().max()) <= .004+2e-6
            break
        assert slew.limited.all(), 'standing but chasing remains limited'
    assert finish_step is not None
    trajectory = torch.stack(targets)
    max_2ms = float((trajectory[1:]-trajectory[:-1]).abs().max())
    max_10ms = float((trajectory[5:]-trajectory[:-5]).abs().max())
    assert max_2ms <= .004+2e-6 and max_10ms <= .02+2e-6
    caught = slew.applied.clone()
    apply_checked(slew)
    assert not slew.limited.any()
    near(slew.applied, caught, 'bypass handoff no jump with unchanged original command')
    pd._control_actions[:, LEGS] = 1.2
    apply_checked(slew)
    near(slew.applied, torch.full((3,4),.6), 'finished follows future original target')
    # Catching before first standing is insufficient to finish.
    raw2, pd2, slew2 = fixture()
    raw2.episode_length_buf.fill_(20)
    apply_checked(slew2)
    apply_checked(slew2)
    assert slew2.limited[:2].all() and not slew2.finished.any()
    return dict(max_2ms_delta_rad=max_2ms, max_10ms_delta_rad=max_10ms,
        rate_rad_s=2., target_reversal=True, standing_alone_does_not_bypass=True,
        catchup_alone_does_not_bypass=True, catchup_handoff_no_jump=True,
        original_targets_after_finish=True)


def check_reset_and_restore():
    raw, pd, slew = fixture()
    raw.episode_length_buf.fill_(20)
    pd.process(torch.arange(18).reshape(3,6)*.02)
    apply_checked(slew)
    slew.started.fill_(True)
    slew.finished[:] = torch.tensor([True,False,True])
    slew.target[:] = torch.arange(12).reshape(3,4)*.01
    before = {name:getattr(slew,name).clone() for name in ('started','finished','target')}
    pd_before = {name:getattr(pd,name).clone() for name in
        ('raw_action','_control_actions','_filtered_leg_actions','_action_history')}
    head, calls = pd._history_head, pd.reset_count
    slew.reset(torch.tensor([1]))
    assert pd.reset_count == calls+1 and pd._history_head == head
    for name, value in before.items():
        actual = getattr(slew,name)
        assert not actual[1].any()
        assert torch.equal(actual[[0,2]],value[[0,2]]), f'partial reset other {name}'
    for name, value in pd_before.items():
        actual = getattr(pd,name)
        if name == '_action_history':
            assert not actual[:,1].any() and torch.equal(actual[:,[0,2]],value[:,[0,2]])
        else:
            assert not actual[1].any() and torch.equal(actual[[0,2]],value[[0,2]])
    raw.sim.data.qpos[1, LEGS] = torch.tensor([.3,.2,-.1,-.2])
    pd.process(torch.ones(3,6)*.4)
    record = apply_checked(slew)
    near(record['target'][1],raw.sim.data.qpos[1, LEGS], 'reset environment relatches measured angles')
    assert slew.finished[[0,2]].all(), 'other finish latches retained'
    slew.reset()
    assert not slew.started.any() and not slew.finished.any() and not slew.target.any()
    assert pd._history_head == (head+1) % len(pd._action_history)
    # Original nominal must also restore if the downstream original raises.
    nominal = pd.nominal_positions.clone()
    pd.raise_on_apply = True
    try:
        slew.apply()
    except RuntimeError as error:
        assert str(error) == 'synthetic original apply failure'
    else:
        raise AssertionError('original apply exception swallowed')
    assert torch.equal(pd.nominal_positions,nominal)
    return dict(partial_reset_only_selected_environment=True, original_reset_called_once=True,
        shared_fifo_head_preserved=True, other_fifo_and_ema_preserved=True,
        selected_environment_relatches=True, full_reset_clears_states=True,
        nominal_restored_on_original_exception=True)


def main(args):
    assert not args.output.exists(), 'output must be a new result file'
    torch.set_num_threads(1)
    result = dict(checks_passed=True, device='cpu', simulation_steps=0, parameter_updates=0,
        hold_release=check_hold_release(), fifo=check_fifo(),
        rate_reverse_handoff=check_rate_reverse_handoff(), reset=check_reset_and_restore(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assessment_sha256=hashlib.sha256((Path(__file__).parent/'assess_getup_target_slew.py').read_bytes()).hexdigest(),
        scope='Synthetic CPU TargetSlew state logic over a minimal original-FIFO/PD double. Does not verify real manager ordering, physics, actual motor response, learned behavior, or production deployment.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
