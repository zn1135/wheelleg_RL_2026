#!/usr/bin/env python3
"""Live LQR-to-student-PD closed-loop diagnostics; no policy or optimizer is created."""
import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction, HuananhuPdActionCfg, wheel_response_input
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.ppo import SmpCommandCfg, inverted, reset_pose
from ct_mjlab.teacher.controller import LqrVmcController
from ct_mjlab.teacher.env import TeacherAction, TeacherActionCfg
from ct_mjlab.teacher.env import body_angles
from ct_mjlab.teacher.standup import FAILED, PHASE_NAMES, FAULT_NAMES
from check_smp_policy_source import balance_substep

ROOT = Path(__file__).resolve().parents[2]
DELAY_PAIRS = ((0., 0.), (.002, .002)) + tuple((a, b) for a in (.006, .008, .010) for b in (.006, .008, .010))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(kw_only=True)
class BridgeCfg(HuananhuPdActionCfg):
    class_type: type = field(default_factory=lambda: ClosedLoopAction)
    bridge_mode: str = 'pd'
    capture_steps: int = 0
    delay_compensation: str = 'none'
    shared_wheel_position_reference: bool = False


class ClosedLoopAction(HuananhuPdAction):
    def synchronize_balance(self, mask):
        """Enter the teacher balance stage only on measured, sustained wheel support."""
        probe = self._env._source_probe
        assert (probe['live'][mask] & (probe['hold'][mask] >= 500)).all()
        assert torch.isfinite(self._env.sim.data.qpos[mask]).all()
        assert torch.isfinite(self._env.sim.data.qvel[mask]).all()
        assert (self._entity.data.projected_gravity_b[mask, 2] < 0).all()
        # 只同步旁路教师阶段和停车参考；不清空学生动作队列或外层滤波历史。
        self.teacher.controller.standup.reset(mask, standing=True)
        if not self.cfg.shared_wheel_position_reference:
            self.teacher.controller.pos[mask] = 0.
        self.teacher.controller.yaw_target[mask] = body_angles(self._entity.data.root_link_quat_w)[2][mask]

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.teacher = TeacherAction(TeacherActionCfg(entity_name='robot', initial='ground',
            velocity_integral=True, ground_start_hold_s=.2), env)
        self.control_dt = env.physics_dt if cfg.bridge_mode == 'torque-500' else env.step_dt
        self.teacher.controller = LqrVmcController(env.num_envs, env.device,
            velocity_integral=True, dt=self.control_dt)
        self.representation_error = torch.zeros(env.num_envs, device=env.device)
        self.target_clipped = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.predictor = None
        self._source_q = torch.zeros_like(self._torques)
        self._source_dq = torch.zeros_like(self._torques)
        self._predicted_dq = torch.zeros_like(self._torques)
        self._substep_ticks = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._generation_ticks = torch.zeros_like(self._substep_ticks)
        self._request_history = torch.zeros_like(self._action_history)
        self._q_history = torch.zeros_like(self._action_history)
        self._dq_history = torch.zeros_like(self._action_history)
        self._prediction_history = torch.zeros_like(self._action_history)
        self._generation_history = torch.full(self._action_history.shape[:2], -1, dtype=torch.long, device=env.device)
        self._capture_index = 0
        self.substeps = {key: torch.empty(cfg.capture_steps, env.num_envs, 6, device=env.device)
            for key in ('q', 'dq', 'source_q', 'source_dq', 'predicted_dq', 'requested', 'actual', 'target', 'actuator_force', 'post_dq')}
        if cfg.capture_steps:
            self.substeps['generation_tick'] = torch.empty(cfg.capture_steps, env.num_envs, 2, dtype=torch.long, device=env.device)
            self.substeps['hold'] = torch.empty(cfg.capture_steps, env.num_envs, dtype=torch.bool, device=env.device)

    def process_actions(self, actions):
        if self.cfg.bridge_mode == 'torque-500':
            super().process_actions(torch.zeros_like(actions))
            return
        reference=getattr(self._env,'_bc_reference_observer',None)
        if reference is None:
            position_reference=None
            if self.cfg.shared_wheel_position_reference:
                from ct_mjlab.envs.chuanliantui.multimode_observations import get_state
                state=get_state(self._env)
                state.synchronize_wheel_handover()
                position_reference=state.wheel_angle_handover_latched
            desired=self.teacher.compute_torques(dt=self.control_dt,position_reference=position_reference)
        else:
            reference.update()
            assert reference.teacher is self.teacher
            desired=self.teacher.torques.clone()
            self.reference_error_used=reference.error.clone()
            self.reference_state_used=reference.state.clone()
            self.reference_events_used=reference.events.clone()
        self.teacher_phase_used=self.teacher.controller.standup.phase.clone()
        self.teacher_fault_used=self.teacher.controller.standup.fault.clone()
        self.teacher_active_used=self.teacher.control_active.clone()
        q = self._env.sim.data.qpos[:, self._motor_q_adr]
        dq = self._env.sim.data.qvel[:, self._motor_v_adr]
        self._source_q.copy_(q)
        self._source_dq.copy_(dq)
        self._predicted_dq.copy_(dq)
        self._generation_ticks.copy_(self._substep_ticks)
        legs, wheels = self._leg_action_ids, self._wheel_action_ids
        # 当前真实反馈反解目标；只在100Hz边界更新，后续完全交给原学生PD/FIFO。
        leg_target = q[:, legs]+(desired[:, legs]+self.cfg.leg_damping*dq[:, legs])/self.cfg.leg_stiffness
        wheel_target = dq[:, wheels]+desired[:, wheels]/self.cfg.wheel_stiffness
        mapped_desired = desired.clone()
        targets = torch.zeros_like(actions)
        targets[:, legs] = (leg_target-self.nominal_positions[:, legs])/self.cfg.leg_position_scale
        targets[:, wheels] = wheel_target/self.cfg.wheel_velocity_scale
        if self.cfg.delay_compensation != 'none':
            predicted_dq, predicted_torque = self.predictor.predict(targets,
                preview_lqr=self.cfg.delay_compensation == 'wheel-lqr')
            # 先补速度变化；完整LQR预览只用于恰好一策略周期的10ms轮延迟。
            active = ~self._ground_hold()
            if self.cfg.delay_compensation == 'wheel-lqr':
                active &= self.wheel_action_delay_steps*self._env.physics_dt >= self._env.step_dt-1e-9
            self._predicted_dq[active] = predicted_dq[active]
            if self.cfg.delay_compensation == 'wheel-lqr':
                mapped_desired[:, wheels] = torch.where(active[:, None], predicted_torque[:, wheels], desired[:, wheels])
            wheel_target = self._predicted_dq[:, wheels]+mapped_desired[:, wheels]/self.cfg.wheel_stiffness
            targets[:, wheels] = wheel_target/self.cfg.wheel_velocity_scale
        self._mapped_desired = mapped_desired
        clipped_leg = leg_target.clamp(self.reference_positions[:, legs]-3.14, self.reference_positions[:, legs]+3.14)
        clipped_wheel = wheel_target.clamp(-self.cfg.max_wheel_velocity, self.cfg.max_wheel_velocity)
        same_state = torch.zeros_like(actions)
        same_state[:, legs] = self.cfg.leg_stiffness*(clipped_leg-q[:, legs])-self.cfg.leg_damping*dq[:, legs]
        same_state[:, wheels] = self.cfg.wheel_stiffness*(clipped_wheel-dq[:, wheels])
        same_state.clamp_(-self._effort_limits, self._effort_limits)
        self.representation_error.copy_(torch.maximum(self.representation_error, (same_state-desired).abs().amax(-1)))
        self.target_clipped += ((clipped_leg != leg_target).any(-1) | (clipped_wheel != wheel_target).any(-1))
        super().process_actions(targets)

    def apply_actions(self):
        super().apply_actions()  # 原学生增益、目标限幅、FIFO、轮端响应和实体弹簧。
        head = self._history_head
        self._request_history[head].copy_(getattr(self, '_mapped_desired', self.teacher.torques))
        self._q_history[head].copy_(self._source_q)
        self._dq_history[head].copy_(self._source_dq)
        self._prediction_history[head].copy_(self._predicted_dq)
        self._generation_history[head].copy_(self._generation_ticks)
        if self.cfg.capture_steps:
            k = self._capture_index
            legs, wheels = self._leg_action_ids, self._wheel_action_ids
            leg_head = (head-self.action_delay_steps) % len(self._action_history)
            wheel_head = (head-self.wheel_action_delay_steps) % len(self._action_history)
            for name, history in (('requested', self._request_history), ('source_q', self._q_history),
                                  ('source_dq', self._dq_history), ('predicted_dq', self._prediction_history)):
                self.substeps[name][k, :, legs] = history[leg_head, self._env_ids][:, legs]
                self.substeps[name][k, :, wheels] = history[wheel_head, self._env_ids][:, wheels]
            self.substeps['q'][k].copy_(self._env.sim.data.qpos[:, self._motor_q_adr])
            self.substeps['dq'][k].copy_(self._env.sim.data.qvel[:, self._motor_v_adr])
            self.substeps['actual'][k].copy_(self.torques)
            delayed = self._action_history[leg_head, self._env_ids]
            self.substeps['target'][k, :, legs] = (self.nominal_positions[:, legs]+self.cfg.leg_position_scale*delayed[:, legs]).clamp(
                self.reference_positions[:, legs]-3.14, self.reference_positions[:, legs]+3.14)
            delayed = self._action_history[wheel_head, self._env_ids]
            self.substeps['target'][k, :, wheels] = (self.cfg.wheel_velocity_scale*delayed[:, wheels]).clamp(
                -self.cfg.max_wheel_velocity, self.cfg.max_wheel_velocity)
            self.substeps['generation_tick'][k, :, 0] = self._generation_history[leg_head, self._env_ids]
            self.substeps['generation_tick'][k, :, 1] = self._generation_history[wheel_head, self._env_ids]
            self.substeps['hold'][k].copy_(self._ground_hold())
            self._capture_index += 1
        if self.cfg.bridge_mode != 'pd':
            if self.cfg.bridge_mode == 'torque-500':
                self.teacher.compute_torques(dt=self.control_dt)
            # 仅诊断组绕过PD目标；绝不把该组结果算作学生接口通过。
            self._torques.copy_(self.teacher.torques.clamp(-self._effort_limits, self._effort_limits))
            effort = self._torques.clone()
            wheels = self._wheel_action_ids
            dq = self._env.sim.data.qvel[:, self._motor_v_adr]
            effort[:, wheels] = wheel_response_input(effort[:, wheels], dq[:, wheels], self._wheel_response)
            self._entity.set_joint_effort_target(effort, joint_ids=self._physical_joint_ids)
        self._substep_ticks.add_(1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, 'teacher'):
            self.teacher.reset(env_ids)
            ids = slice(None) if env_ids is None else env_ids
            self.teacher.command[ids] = self.teacher.command.new_tensor([0., 0., .30])
            self.representation_error[ids] = 0
            self.target_clipped[ids] = 0
            for value in (self._source_q, self._source_dq, self._predicted_dq, self._substep_ticks, self._generation_ticks):
                value[ids] = 0
            for value in (self._request_history, self._q_history, self._dq_history, self._prediction_history):
                value[:, ids] = 0
            self._generation_history[:, ids] = -1


def teacher_failed(env):
    return env.action_manager.get_term('pd').teacher.controller.standup.phase == FAILED


def capture_applied_force(env):
    action = env.action_manager.get_term('pd')
    if action.cfg.capture_steps and action._capture_index:
        index = action._capture_index-1
        # mj_step保留本物理步实际使用的执行器力；dq已是积分后的状态。
        action.substeps['actuator_force'][index].copy_(env.sim.data.actuator_force[:, action.teacher.motor_actuators])
        action.substeps['post_dq'][index].copy_(env.sim.data.qvel[:, action._motor_v_adr])
    return torch.zeros(env.num_envs, device=env.device)


@torch.inference_mode()
def run(args):
    dataset = MotionDataset(args.dataset)
    assert dataset.xml_sha256 == digest(MODEL_XML)
    episodes = [row for row in dataset.episodes if row['behavior'] == 'standup']
    records = []
    for row in episodes:
        with np.load(row['path']) as trace:
            records.append((trace['initial_qpos'].copy(), trace['initial_qvel'].copy()))
    pairs = DELAY_PAIRS if args.mode == 'pd' else ((0., 0.),)
    if args.training_delays or args.stress_10ms:
        assert args.mode == 'pd', 'Training delays certify only the student PD bridge'
        training_cfg, _ = task_registry.get_cfgs('chuanliantui', play=False)
        choices = training_cfg.actions['pd'].action_delay_choices_s
        if args.stress_10ms:
            choices = tuple(sorted(set(choices) | {.010}))
        pairs = tuple((leg, wheel) for leg in choices for wheel in choices)
        if args.stress_10ms:
            pairs = tuple(pair for pair in pairs if .010 in pair)
    count = len(episodes)*len(pairs)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    cfg.seed, cfg.auto_reset = 42, False
    cfg.commands['standup'] = SmpCommandCfg()
    cfg.events['reset_pose'] = EventTermCfg(func=reset_pose, mode='reset')
    cfg.terminations['inverted'] = TerminationTermCfg(func=inverted)
    cfg.terminations['teacher_failed'] = TerminationTermCfg(func=teacher_failed)
    cfg.rewards = {}
    cfg.metrics['source_balance'] = MetricsTermCfg(func=balance_substep, per_substep=True, reduce='last')
    cfg.metrics['capture_force'] = MetricsTermCfg(func=capture_applied_force, per_substep=True, reduce='last')
    cfg.actions['pd'] = BridgeCfg(**{key: value for key, value in vars(cfg.actions['pd']).items() if key != 'class_type'},
                                 bridge_mode=args.mode, capture_steps=round(args.seconds/.002) if args.high_rate else 0,
                                 delay_compensation=args.compensation)
    cfg.actions['pd'].action_delay_s = None
    cfg.actions['pd'].randomize_delays = False
    cfg.actions['pd'].action_delay_choices_s = tuple(sorted({value for pair in pairs for value in pair}))
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                    num_envs=count, device=args.device)
    try:
        raw = env.unwrapped
        action = raw.action_manager.get_term('pd')
        raw.reset(seed=42)
        if args.compensation != 'none':
            from lqr_delay_predictor import QueuedPhysicsPredictor
            action.predictor = QueuedPhysicsPredictor(raw, action)
        qpos = torch.tensor(np.asarray([q for pair in pairs for q, v in records]), device=args.device)
        qvel = torch.tensor(np.asarray([v for pair in pairs for q, v in records]), device=args.device)
        # 完整根/闭链姿态与示范一致，只把世界XY平移到各环境原点。
        qpos[:, :2] = raw.scene.env_origins[:, :2]
        raw.sim.data.qpos.copy_(qpos)
        raw.sim.data.qvel.copy_(qvel)
        raw.sim.forward()
        leg_delay = torch.tensor([round(a/.002) for a, b in pairs], device=args.device).repeat_interleave(len(episodes))
        wheel_delay = torch.tensor([round(b/.002) for a, b in pairs], device=args.device).repeat_interleave(len(episodes))
        action.action_delay_steps.copy_(leg_delay)
        action.wheel_action_delay_steps.copy_(wheel_delay)
        initial_qpos, initial_qvel = qpos.cpu().numpy(), qvel.cpu().numpy()
        raw._source_weight = float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        raw._source_probe = dict(live=torch.ones(count, dtype=torch.bool, device=args.device),
            hold=torch.zeros(count, dtype=torch.long, device=args.device),
            strict_run=torch.zeros(count, dtype=torch.long, device=args.device),
            strict_ever=torch.zeros(count, dtype=torch.bool, device=args.device))
        probe, reader = raw._source_probe, MotionStateReader(raw)
        buffers = {key: [] for key in ('root', 'qpos', 'qvel', 'motor_q', 'motor_dq', 'features',
            'wheel_position_w', 'ground_height',
            'torque', 'requested_torque', 'action', 'alive', 'phase', 'fault', 'done', 'time_out')}
        steps = round(args.seconds/.01)
        for tick in range(steps):
            live = probe['live'].clone()
            _, _, done, _ = env.step(torch.zeros(count, 6, device=args.device))
            state = reader.read()
            values = dict(root=state['root'], qpos=raw.sim.data.qpos, qvel=raw.sim.data.qvel,
                wheel_position_w=state['wheel_position_w'], ground_height=state['ground_height'],
                motor_q=state['motor_q'], motor_dq=state['motor_dq'], features=reader.features(),
                torque=action.torques, requested_torque=action.teacher.torques,
                action=action.raw_action, alive=live, phase=action.teacher.controller.standup.phase,
                fault=action.teacher.controller.standup.fault, done=done.bool(), time_out=raw.reset_time_outs)
            for key, value in values.items():
                buffers[key].append(value.cpu().numpy().copy())
            if tick < 20:
                assert not action.torques.any()
            assert (raw.command_manager.get_term('standup').command[:, :2] == 0.).all()
            probe['live'] &= ~done.bool()
            if tick % 500 == 0:
                print(f'{args.mode}: t={(tick+1)*.01:.2f}s live={int(probe["live"].sum())}/{count}', flush=True)
            if not probe['live'].any():
                break
            if done.any():
                raw.reset(env_ids=done.nonzero().flatten())
                action.action_delay_steps.copy_(leg_delay)
                action.wheel_action_delay_steps.copy_(wheel_delay)
        arrays = {key: np.asarray(value) for key, value in buffers.items()}
        times = (np.arange(len(arrays['root']))+1)*.01
        np.savez_compressed(args.output/'trace.npz', **arrays, time=times,
                            initial_qpos=initial_qpos, initial_qvel=initial_qvel)
        if args.high_rate:
            np.savez_compressed(args.output/'substeps.npz',
                **{key: value[:action._capture_index].cpu().numpy() for key, value in action.substeps.items()},
                time=np.arange(action._capture_index)*raw.physics_dt)
        rows = []
        for index in range(count):
            length = int(arrays['alive'][:, index].sum())
            final = max(0, length-1)
            complete = length == 2000 and bool(arrays['time_out'][final, index])
            late = (times[:length] > 15.) & (times[:length] <= 20.)
            root = arrays['root'][:length, index]
            legs = arrays['motor_dq'][:length, index][:, [0, 1, 3, 4]]
            excursion = float(np.linalg.norm(root[late, :2]-root[late, :2][0], axis=1).max()) if late.any() else None
            speed = float(np.linalg.norm(root[late, 7:9], axis=1).mean()) if late.any() else None
            rms = float(np.sqrt(np.square(legs[late]).mean())) if late.any() else None
            finite = bool(np.isfinite(arrays['qpos'][:length, index]).all())
            row = dict(env=index, episode_id=episodes[index % len(episodes)]['episode_id'],
                teacher_episode_path=str(episodes[index % len(episodes)]['path']),
                leg_delay_s=pairs[index//len(episodes)][0], wheel_delay_s=pairs[index//len(episodes)][1],
                observed_s=round(length*.01, 2), full_20s=complete, strict_rise=bool(probe['strict_ever'][index]),
                final_balance_v31=bool(complete and probe['hold'][index] >= 500), finite=finite,
                final_phase=PHASE_NAMES[int(arrays['phase'][final, index])],
                final_fault=FAULT_NAMES[int(arrays['fault'][final, index])],
                final_height_m=float(arrays['features'][final, index, 0]),
                last5s_xy_excursion_m=excursion, last5s_mean_speed_m_s=speed,
                last5s_leg_velocity_rms_rad_s=rms,
                max_same_state_inverse_pd_error_nm=(float(action.representation_error[index]) if args.compensation == 'none' else None),
                max_preview_torque_change_at_generation_nm=(float(action.representation_error[index]) if args.compensation != 'none' else None),
                clipped_target_policy_steps=int(action.target_clipped[index]))
            rows.append(row)
        sources = list((ROOT/'mjlab_training/src/ct_mjlab/teacher').glob('*.py'))+[Path(__file__),
            ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_action.py',
            ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/multimode_env_cfg.py',
            ROOT/'scripts/agent/check_smp_policy_source.py']
        if args.compensation != 'none':
            sources.append(ROOT/'scripts/agent/lqr_delay_predictor.py')
        report = dict(completed=True, mode=args.mode, compensation=args.compensation, optimizer_updates=0, num_envs=count,
            physics_dt=.002, action_dt=.01, controller_dt=action.control_dt, ground_start_hold_s=.2,
            xml_sha256=dataset.xml_sha256, dataset_fingerprint=dataset.fingerprint,
            delay_pairs_s=pairs, training_delay_grid=args.training_delays,
            stress_delay_ms=10 if args.stress_10ms else None,
            rows=rows, trace_sha256=digest(args.output/'trace.npz'),
            source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
            prediction_calls=action.predictor.calls if action.predictor is not None else 0,
            preview_note='Compensation intentionally changes same-state torque at generation time; correctness is checked at actual execution time. Preview uses a separate nominal model and current full state/queued commands, not future real observations.',
            scope='Live privileged teacher feedback, not replayed targets; nominal play motors. No training, observation noise, motor randomization or pushes. Retimed derivative/ramp/timers/filter; Q scaled by dt/1ms is an approximation. Torque modes are isolation diagnostics and do not certify the student PD interface.')
        (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml', asdict(cfg))
        print(json.dumps(dict(mode=args.mode, count=count,
            strict_rise=sum(r['strict_rise'] for r in rows),
            final_balance=sum(r['final_balance_v31'] for r in rows)), indent=2), flush=True)
    finally:
        if 'action' in locals() and action.predictor is not None:
            action.predictor.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--mode', choices=('pd', 'torque-100', 'torque-500'), default='pd')
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--high-rate', action='store_true', help='Capture the exact pre-physics 500Hz FIFO/PD path')
    delays = parser.add_mutually_exclusive_group()
    delays.add_argument('--training-delays', action='store_true', help='Use the current training delay Cartesian grid')
    delays.add_argument('--stress-10ms', action='store_true', help='Test pairs with a 10ms leg or wheel delay; do not change training')
    parser.add_argument('--compensation', choices=('none', 'wheel-velocity', 'wheel-lqr'), default='none')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
