#!/usr/bin/env python3
"""Frozen 500 Hz yaw-rate/heading-hold measurement; separate from balance v3.1."""

from scripts.agent.motion.assess import STANDARD, THRESHOLDS as MOTION_THRESHOLDS, stand_phase, settled_window

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import mujoco
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML, ACTUATED_JOINTS
from ct_mjlab.envs.chuanliantui_stand_v2 import mdp
from ct_mjlab.envs.chuanliantui_stand_v2.observations import get_state
from ct_mjlab.envs.chuanliantui_stand_v2.rewards import configure_tight_reward, restore_tracking_reward_cfg
from scripts.agent.stand_v2.assess_chuanliantui_stand_v2 import FIELDS, Recorder, digest, sample_physics, summarize
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import tensor_digest


GROUPS = tuple((vx, yaw) for vx in (0., .3, -.3) for yaw in (0., .3, -.3, .5, -.5))
SEGMENTS = ((8., 14., "rate", 1., 1.), (14., 20., "heading_hold", 1., 0.),
            (20., 26., "reverse_rate", 1., -1.), (26., 32., "heading_hold", 1., 0.),
            (32., 40., "stop_heading_hold", 0., 0.))
COMMAND_FIELDS = ("request_vx_m_s", "request_yaw_rad_s", "command_vx_m_s", "command_yaw_rad_s",
                  "body_vx_m_s", "body_yaw_rad_s", "heading_rad", "heading_reference_rad",
                  "heading_hold", "vx_gate", "yaw_gate", "has_stood", "recover", "fallen", "ready_steps")
THRESHOLDS = dict(yaw_mae_rad_s=MOTION_THRESHOLDS["yaw_mae_rad_s"], vx_mae_m_s=MOTION_THRESHOLDS["vx_mae_m_s"], heading_final1_max_deg=5.,
                  stopped_planar_speed_m_s=MOTION_THRESHOLDS["parking_horizontal_peak_m_s"], steady_wait_after_ramp_s=.5,
                  steady_min_measurement_s=.5, command_target_tolerance=1e-5, final_window_s=1.)

POSITION_FIELDS = ('locked', 'ready_steps', 'reference_x', 'reference_y', 'forward_error', 'lateral_error',
    'correction_request', 'command_vx', 'true_forward_error', 'true_lateral_error', 'odom_x', 'odom_y',
    'true_x', 'true_y', 'odom_minus_truth_m', 'wheel_vx', 'odom_relative_bias_since_anchor_m')


def wheel_geometry(model, prefix=''):
    """Read joint order/axis and collision radius from model.py's compiled XML."""
    rows = []
    for name in (ACTUATED_JOINTS[2], ACTUATED_JOINTS[5]):
        joint = model.joint(prefix+name).id
        axis = model.jnt_axis[joint]
        if abs(axis[1]) < .99:
            raise ValueError(f'Wheel axis is not body Y: {name}')
        radii = []
        for geom in np.flatnonzero(model.geom_bodyid == model.jnt_bodyid[joint]):
            mesh = model.geom_dataid[geom]
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]) or mesh < 0:
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError(f'Expected wheel collision mesh: {name}')
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            vertices = model.mesh_vert[first:first+count] @ rotation.reshape(3,3).T
            offset = vertices + model.geom_pos[geom] - model.jnt_pos[joint]
            radial = offset - np.outer(offset @ axis, axis)
            radii.append(float(np.linalg.norm(radial, axis=1).max()))
        if not radii:
            raise ValueError(f'No wheel collision radius: {name}')
        rows.append(dict(joint=name, model_joint=prefix+name, radius_m=max(radii), forward_sign=float(np.sign(axis[1]))))
    return rows


def integrate_wheels(xy, wheel_vx, heading, dt):
    return xy + torch.stack((heading.cos(), heading.sin()), -1) * wheel_vx[:, None] * dt


class YawRecorder(Recorder):
    def __init__(self, raw):
        super().__init__(raw)
        samples = round(40. / raw.physics_dt)
        self.trace = torch.empty((samples, raw.num_envs, len(FIELDS)), device=raw.device)
        self.command_trace = torch.empty((samples, raw.num_envs, len(COMMAND_FIELDS)), device=raw.device)
        self.requests = torch.zeros((raw.num_envs, 2), device=raw.device)

    def sample(self):
        i = self.count
        super().sample()
        q = self.raw.sim.data.qpos[:, self.root_qadr]
        v = self.raw.sim.data.qvel[:, self.root_vadr]
        w, x, y, z = q[:, 3:].unbind(-1)
        vx = ((1-2*(y*y+z*z))*v[:, 0] + 2*(x*y+w*z)*v[:, 1] + 2*(x*z-w*y)*v[:, 2])
        heading = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        s = self.raw.command_manager.get_term("standup")
        # MuJoCo自由关节的平移速度在世界系，角速度在机体系。
        self.command_trace[i] = torch.stack((self.requests[:, 0], self.requests[:, 1],
            s.command[:, 0], s.command[:, 1], vx, v[:, 5], heading, s.heading_target,
            s.heading_hold.float(), s.tracking_active.float(), s.yaw_active.float(),
            s.has_stood.float(), s.recover.float(), s.fallen.float(), s.tracking_ready_steps.float()), -1)


class ParkingRecorder(YawRecorder):
    def __init__(self, raw):
        super().__init__(raw)
        self.position_trace = torch.empty((len(self.trace), raw.num_envs, len(POSITION_FIELDS)), device=raw.device)
        self.geometry = wheel_geometry(raw.sim.mj_model, prefix='robot/')
        ids, _ = raw.scene['robot'].find_joints(tuple(row['joint'] for row in self.geometry), preserve_order=True)
        self.wheel_vadr = raw.scene['robot'].indexing.joint_v_adr[ids]
        self.wheel_scale = torch.tensor([row['radius_m']*row['forward_sign'] for row in self.geometry], device=raw.device)
        self.odom_xy = raw.sim.data.qpos[:, self.root_qadr][:, :2].clone()
        self.wheel_vx = torch.zeros(raw.num_envs, device=raw.device)
        self.true_anchor = torch.zeros_like(self.odom_xy)
        self.odom_anchor = torch.zeros_like(self.odom_xy)
        self.was_locked = torch.zeros_like(self.live)
        self.position_source = 'truth'

    def after_command(self):
        s = self.raw.command_manager.get_term('standup')
        lock = s.position_locked & ~self.was_locked
        self.true_anchor[lock] = self.raw.scene['robot'].data.root_link_pos_w[lock, :2]
        self.odom_anchor[lock] = self.odom_xy[lock]
        self.was_locked.copy_(s.position_locked)

    def sample(self):
        index = self.count
        super().sample()
        c = self.command_trace[index]
        self.wheel_vx.copy_((self.raw.sim.data.qvel[:, self.wheel_vadr] * self.wheel_scale).mean(-1))
        self.odom_xy.copy_(integrate_wheels(self.odom_xy, self.wheel_vx, c[:, 6], self.raw.physics_dt))
        s = self.raw.command_manager.get_term('standup')
        xy = self.raw.sim.data.qpos[:, self.root_qadr][:, :2]
        offset = xy-self.true_anchor
        heading = c[:, 6]
        forward = offset[:, 0]*heading.cos()+offset[:, 1]*heading.sin()
        lateral = -offset[:, 0]*heading.sin()+offset[:, 1]*heading.cos()
        measured_offset = (self.odom_xy if self.position_source == 'wheel-odometry' else xy)-s.position_anchor
        measured_forward = measured_offset[:, 0]*heading.cos()+measured_offset[:, 1]*heading.sin()
        measured_lateral = -measured_offset[:, 0]*heading.sin()+measured_offset[:, 1]*heading.cos()
        self.position_trace[index] = torch.stack((s.position_locked.float(), s.position_ready_steps.float(),
            s.position_anchor[:, 0], s.position_anchor[:, 1], measured_forward, measured_lateral,
            s.position_velocity, s.command[:, 0], forward, lateral, self.odom_xy[:, 0], self.odom_xy[:, 1],
            xy[:, 0], xy[:, 1], (self.odom_xy-xy).norm(dim=-1), self.wheel_vx,
            ((self.odom_xy-self.odom_anchor)-(xy-self.true_anchor)).norm(dim=-1)), -1)


def position_metrics(p, c, trace, dt, tolerance, *, terminated=False):
    valid = (p[:, 12] > .5) & (p[:, 13] > .5) & np.isfinite(trace).all(-1) & np.isfinite(c).all(-1)
    locked = valid & (trace[:, 0] > .5)
    tail = np.arange(len(trace)) >= len(trace)-round(1./dt)
    final_complete = bool(tail.any() and locked[tail].all() and not terminated)
    def errors(mask, forward, lateral):
        if not mask.any():
            return None
        return dict(forward_mean_abs_m=float(np.abs(trace[mask, forward]).mean()),
            forward_max_abs_m=float(np.abs(trace[mask, forward]).max()),
            forward_final_m=float(trace[np.flatnonzero(mask)[-1], forward]),
            lateral_mean_abs_m=float(np.abs(trace[mask, lateral]).mean()),
            lateral_max_abs_m=float(np.abs(trace[mask, lateral]).max()))
    def crossings(column):
        count, previous = 0, 0
        for live, value in zip(locked, trace[:, column]):
            if not live:
                previous = 0
            elif abs(value) > tolerance:
                sign = int(np.sign(value))
                count += int(previous != 0 and sign != previous)
                previous = sign
        return count
    def maximum(values):
        return float(np.maximum(values[locked], 0.).max()) if locked.any() else None
    return dict(terminated_by_segment_end=terminated, valid_s=float(valid.sum()*dt), locked_s=float(locked.sum()*dt),
        unlocked_s=float((valid & ~locked).sum()*dt),
        first_locked_offset_s=float((np.flatnonzero(locked)[0]+1)*dt) if locked.any() else None,
        final1_complete=final_complete,
        whole_locked_measured=errors(locked,4,5), whole_locked_truth=errors(locked,8,9),
        final1_measured=errors(tail & locked,4,5) if final_complete else None,
        final1_truth=errors(tail & locked,8,9) if final_complete else None,
        max_return_request_m_s=maximum(-np.sign(trace[:,4])*trace[:,6]),
        max_return_command_m_s=maximum(-np.sign(trace[:,4])*trace[:,7]),
        max_true_return_velocity_m_s=maximum(-np.sign(trace[:,8])*c[:,4]),
        measured_error_zero_crossings=crossings(4), true_error_zero_crossings=crossings(8),
        oscillation_candidate=crossings(8)>=2,
        odometry_bias_final_m=float(trace[np.flatnonzero(valid)[-1],14]) if valid.any() else None,
        odometry_bias_max_m=float(trace[valid,14].max()) if valid.any() else None,
        odometry_relative_bias_final_m=float(trace[np.flatnonzero(locked)[-1],16]) if locked.any() else None,
        odometry_relative_bias_max_m=float(trace[locked,16].max()) if locked.any() else None)


def wrapped_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def segment_metrics(p, c, *, start, end, kind, vx, yaw, previous_yaw, dt, episode_end_s=None, parking=False):
    """固定时间窗保留全部样本分母；复位后的物理/命令数据不进入统计。"""
    live = (p[:, 13] > .5)
    finite = (p[:, 12] > .5) & np.isfinite(c).all(-1)
    valid = live & finite
    gated = valid & (c[:, 9] > .5) & (c[:, 10] > .5)
    terminated = episode_end_s is not None and episode_end_s <= end
    full = bool(gated.all() and not terminated)
    is_hold = kind in ("heading_hold", "stop_heading_hold") or yaw == 0.
    settled = gated.copy() if parking else gated & (np.abs(c[:, 2]-vx) <= THRESHOLDS["command_target_tolerance"])
    if not is_hold:
        settled &= np.abs(c[:, 3]-yaw) <= THRESHOLDS["command_target_tolerance"]
    # 必须完成实际限斜率，再连续等0.5秒；慢斜率不能仅因跟随中间命令而通过。
    measured = settled_window(settled, dt, THRESHOLDS["steady_wait_after_ramp_s"])
    measured_full = bool(measured.sum()*dt >= THRESHOLDS["steady_min_measurement_s"])
    yaw_mae = float(np.abs(c[measured, 5]-c[measured, 3]).mean()) if measured.any() else None
    vx_mae = float(np.abs(c[measured, 4]-c[measured, 2]).mean()) if measured.any() else None
    requested_yaw_mae = float(np.abs(c[measured, 5]-yaw).mean()) if measured.any() else None
    requested_vx_mae = float(np.abs(c[measured, 4]-vx).mean()) if measured.any() else None
    angle = wrapped_angle(c[:, 6]-c[:, 7])
    holds = gated & (c[:, 8] > .5)
    full_hold = bool(holds.all() and not terminated)
    tail = np.arange(len(c)) >= len(c)-round(THRESHOLDS["final_window_s"] / dt)
    hold_complete = bool(holds[tail].all() and tail.any())
    final_error = float(np.rad2deg(np.abs(angle[tail]).max())) if hold_complete else None
    peak_error = float(np.rad2deg(np.abs(angle[holds]).max())) if holds.any() else None
    # 顺着松杆前转向方向继续转过锁定点的最大角度；零转向对照组没有方向。
    overshoot = (float(np.rad2deg(np.maximum(angle[holds]*np.sign(previous_yaw), 0.).max()))
                 if holds.any() and previous_yaw != 0. else None)
    speed = np.linalg.norm(p[:, 7:9], axis=-1)
    tail_valid = bool(valid[tail].all() and tail.any())
    stop_speed = float(speed[tail].max()) if tail_valid else None
    xy = p[valid, 9:11]
    tilt = np.rad2deg(np.arccos((-p[:, 14]).clip(-1., 1.)))
    motion = stand_phase(p, c, selected=measured, vx_target=vx, yaw_target=yaw,
        complete=bool(full and measured_full), yaw_columns=True)
    return dict(motion_standard=motion, kind=kind, start_s=start, end_s=end, requested_vx_m_s=vx, requested_yaw_rad_s=yaw,
        terminated_by_segment_end=terminated,
        full_live_finite_and_gated=full, live_s=float(live.sum()*dt), measured_s=float(measured.sum()*dt),
        steady_window_complete=measured_full, yaw_mae_to_command_rad_s=yaw_mae,
        yaw_mae_to_request_rad_s=requested_yaw_mae, vx_mae_to_command_m_s=vx_mae,
        vx_mae_to_request_m_s=requested_vx_mae,
        response_yaw_mae_to_command_rad_s=float(np.abs(c[gated, 5]-c[gated, 3]).mean()) if gated.any() else None,
        response_yaw_mae_to_request_rad_s=float(np.abs(c[gated, 5]-yaw).mean()) if gated.any() else None,
        yaw_rate_indicator=bool(motion["checks"]["complete"] and motion["checks"]["tracked_yaw"]) if not is_hold else None,
        vx_indicator=bool(motion["checks"]["complete"] and motion["checks"]["tracked_vx"]),
        heading_hold_expected=is_hold, full_heading_hold=full_hold, final1_heading_hold_complete=hold_complete,
        final1_max_heading_error_deg=final_error, peak_heading_error_deg=peak_error,
        peak_directional_overshoot_deg=overshoot,
        heading_indicator=bool(full_hold and hold_complete and final_error <= THRESHOLDS["heading_final1_max_deg"]) if is_hold else None,
        planar_drift_indicator=bool(full and tail_valid and stop_speed <= THRESHOLDS["stopped_planar_speed_m_s"]) if vx == 0. else None,
        final1_max_planar_speed_m_s=stop_speed,
        mean_planar_speed_m_s=float(speed[valid].mean()) if valid.any() else None,
        xy_excursion_m=float(np.linalg.norm(xy-xy[0], axis=-1).max()) if len(xy) else None,
        xy_path_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum()) if len(xy) else None,
        max_tilt_deg=float(tilt[valid].max()) if valid.any() else None,
        body_contact_s=float(((p[:, 2] > 1.) & valid).sum()*dt),
        inverted=bool(((p[:, 14] > 0.) & valid).any()),
        ungated_s=float((valid & ~gated).sum()*dt), nonfinite_s=float((live & ~finite).sum()*dt))


def indicator_counts(rows):
    result = {}
    for name in ("yaw_rate_indicator", "vx_indicator", "heading_indicator", "planar_drift_indicator"):
        values = [s[name] for row in rows for s in row["segments"] if s[name] is not None]
        result[name] = dict(total=len(values), passed=sum(values), fraction=sum(values)/len(values) if values else None)
    return result


def selfcheck():
    p = np.zeros((3000, len(FIELDS))); c = np.zeros((3000, len(COMMAND_FIELDS)))
    p[:, 12:14] = 1.; p[:, 14] = -1.; c[:, 8:12] = 1.
    call = lambda: segment_metrics(p, c, start=14., end=20., kind="heading_hold",
                                   vx=0., yaw=0., previous_yaw=.3, dt=.002)
    c[:, 6] = np.deg2rad(-179.); c[:, 7] = np.deg2rad(179.)
    good = call()
    assert good["heading_indicator"] and abs(good["final1_max_heading_error_deg"]-2.) < 1e-9
    assert abs(good["peak_directional_overshoot_deg"]-2.) < 1e-9
    c[:, 9:11] = 0.
    never_ready = call()
    assert not never_ready["yaw_rate_indicator"] and never_ready["yaw_mae_to_command_rad_s"] is None
    assert not never_ready["heading_indicator"] and never_ready["final1_max_heading_error_deg"] is None
    c[:, 9:11] = 1.; p[500:, 13] = 0.; p[500:, 9:11] = 1000.; c[500:, 6] = 0.
    failed = call()
    assert not failed["heading_indicator"] and failed["final1_max_planar_speed_m_s"] is None
    assert failed["xy_excursion_m"] == 0. and failed["live_s"] == 1.
    counts = indicator_counts([dict(segments=[good, never_ready, failed])])
    assert counts["heading_indicator"] == dict(total=3, passed=1, fraction=1/3)
    p[:, 13] = 1.; c[:, 6:8] = 0.; c[:, 5] = .1001
    assert call()["yaw_rate_indicator"] is None and call()["yaw_mae_to_command_rad_s"] > .1
    c[:, 5] = 0.; c[:, 6] = np.deg2rad(5.001)
    assert not call()["heading_indicator"]
    c[:, 6] = 0.; p[-1, 12] = 0.
    assert not call()["heading_indicator"]
    p[-1, 12] = 1.
    boundary = segment_metrics(p, c, start=14., end=20., kind="heading_hold", vx=0.,
        yaw=0., previous_yaw=.3, dt=.002, episode_end_s=20.)
    assert not boundary["heading_indicator"] and not boundary["yaw_rate_indicator"]
    c[:, :8] = 0.; c[:, 0] = .3; c[:, 1] = .5; c[:, 2] = c[:, 4] = .3
    c[:, 8] = 0.
    rate = lambda: segment_metrics(p, c, start=8., end=14., kind="rate", vx=.3,
                                   yaw=.5, previous_yaw=0., dt=.002)
    c[:, 3] = c[:, 5] = np.linspace(0., .3, len(c))
    slow = rate()
    assert not slow["yaw_rate_indicator"] and slow["measured_s"] == 0.
    assert slow["yaw_mae_to_command_rad_s"] is None
    assert slow["response_yaw_mae_to_command_rad_s"] == 0.
    assert abs(slow["response_yaw_mae_to_request_rad_s"]-.35) < 1e-9
    c[:, 3] = c[:, 5] = .5
    c[:2550, 3] = c[:2550, 5] = .4
    too_short = rate()
    assert abs(too_short["measured_s"]-.4) < 1e-9 and not too_short["yaw_rate_indicator"]
    c[:, 3] = c[:, 5] = .5
    c[:2500, 3] = c[:2500, 5] = .4
    assert rate()["measured_s"] == .5 and rate()["yaw_rate_indicator"]
    c[:, 2] = c[:, 4] = .2
    assert rate()["measured_s"] == 0. and not rate()["vx_indicator"]
    c[:, 2] = c[:, 4] = .3; c[:, 3] = .5; c[:, 5] = .6001
    assert not rate()["yaw_rate_indicator"]
    assert indicator_counts([])["heading_indicator"]["fraction"] is None
    geometry = wheel_geometry(mujoco.MjModel.from_xml_path(str(MODEL_XML)))
    assert [row['joint'] for row in geometry] == [ACTUATED_JOINTS[2], ACTUATED_JOINTS[5]]
    assert [row['forward_sign'] for row in geometry] == [1., -1.]
    assert all(.059 < row['radius_m'] < .061 for row in geometry)
    xy = torch.zeros((2,2)); velocity = torch.ones(2); heading = torch.tensor([0., np.pi/2])
    traveled = integrate_wheels(xy, velocity, heading, 1.)
    torch.testing.assert_close(traveled, torch.tensor([[1.,0.],[0.,1.]]), atol=1e-6, rtol=0)
    scale = np.asarray([row['radius_m']*row['forward_sign'] for row in geometry])
    assert abs(np.mean((1./scale)*scale)-1.) < 1e-9
    assert abs(np.mean((np.asarray([1.,-1.])/scale)*scale)) < 1e-9
    physical = np.zeros((1000,len(FIELDS))); command = np.zeros((1000,len(COMMAND_FIELDS)))
    trace = np.zeros((1000,len(POSITION_FIELDS))); physical[:,12:14] = 1.; trace[:,0] = 1.
    trace[:,4] = trace[:,8] = .03; trace[300:600,4] = trace[300:600,8] = -.03
    trace[:,6:8] = -np.sign(trace[:,4,None])*.03; trace[:,14] = .7
    parking = position_metrics(physical,command,trace,.002,.02)
    assert parking['true_error_zero_crossings']==2 and parking['oscillation_candidate']
    assert abs(parking['whole_locked_truth']['forward_max_abs_m']-.03)<1e-9
    assert parking['odometry_bias_final_m']==.7 and parking['max_return_request_m_s']==.03
    assert position_metrics(physical,command,trace,.002,.02,terminated=True)['final1_truth'] is None
    trace[-500:,0] = 0.
    assert position_metrics(physical,command,trace,.002,.02)['final1_truth'] is None
    trace[:,0] = 0.
    assert position_metrics(physical,command,trace,.002,.02)['whole_locked_truth'] is None
    physical[:,13] = 0.; trace[:,0] = 1.
    assert position_metrics(physical,command,trace,.002,.02)['locked_s']==0.
    print("Yaw measurement selfcheck passed: angle wrap, not-ready, first-episode freeze, fixed denominators, slow slew/request errors; no simulation")
    print('Parking selfcheck passed: model wheel axes/radii, signed travel/rotation, projection, slip bias, dead-zone crossings and missing-lock windows')


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    source_bytes = Path(__file__).read_bytes()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    infos = saved["infos"]
    contract = infos.get("command_contract", mdp.ZERO_COMMAND_CONTRACT)
    native_position = contract == mdp.POSITION_COMMAND_CONTRACT
    native = contract in (mdp.YAW_COMMAND_CONTRACT, mdp.POSITION_COMMAND_CONTRACT)
    position_enabled = native_position or args.probe_position_hold
    if args.probe_position_hold and not (native or args.probe_yaw):
        raise ValueError('--probe-position-hold requires native yaw or --probe-yaw')
    if args.position_source != 'truth' and not position_enabled:
        raise ValueError('Wheel odometry requires position hold')
    if args.truth_lock_speed and args.position_source != 'wheel-odometry':
        raise ValueError('--truth-lock-speed is only a wheel-position diagnostic')
    if not native and not args.probe_yaw:
        raise ValueError("Non-yaw checkpoint requires --probe-yaw: frozen capability probe only")
    if contract not in (mdp.ZERO_COMMAND_CONTRACT, mdp.TRACKING_COMMAND_CONTRACT,
                         mdp.SPEED_COMMAND_CONTRACT, mdp.YAW_COMMAND_CONTRACT, mdp.POSITION_COMMAND_CONTRACT):
        raise ValueError(f"Unsupported command contract: {contract}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source.py').write_bytes(source_bytes)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2", play=args.conditions == "nominal")
    cfg.auto_reset = False
    cfg.metrics["yaw_physics"] = MetricsTermCfg(func=sample_physics, per_substep=True, reduce="last")
    # 初态仍走原零速路径；先按模型原契约加载，再显式打开旧模型能力探测。
    env, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        num_envs=args.num_envs, seed=args.seed, device=args.device)
    try:
        raw, s = env.unwrapped, env.unwrapped.command_manager.get_term("standup")
        s.cfg.post_stand_tracking = contract != mdp.ZERO_COMMAND_CONTRACT
        s.cfg.speed_curriculum = contract in (mdp.SPEED_COMMAND_CONTRACT, mdp.YAW_COMMAND_CONTRACT, mdp.POSITION_COMMAND_CONTRACT)
        s.cfg.yaw_control = native
        s.cfg.position_hold = native_position
        restore_tracking_reward_cfg(s.cfg, infos)
        if s.cfg.tracking_tight_term:
            from mjlab.managers.reward_manager import RewardManager
            configure_tight_reward(raw.cfg, s.cfg)
            raw.reward_manager = RewardManager(raw.cfg.rewards, raw, scale_by_dt=raw.cfg.scale_rewards_by_dt)
        if s.cfg.speed_curriculum:
            for key, value in infos["standup_curriculum"]["speed_curriculum"]["config"].items():
                setattr(s.cfg, key, value)
        if native:
            for key, value in infos["yaw_control_config"].items():
                setattr(s.cfg, key, value)
        if native_position:
            for key, value in infos['position_hold_config'].items():
                setattr(s.cfg, key, value)
        runner, _ = task_registry.make_alg_runner(env, "chuanliantui_stand_v2", train_cfg=agent, device=args.device)
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No learning")):
            runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
            frozen = tensor_digest(runner.alg.actor.state_dict())
            s.cfg.post_stand_tracking = s.cfg.speed_curriculum = s.cfg.yaw_control = True
            s.cfg.position_hold = position_enabled
            if position_enabled:
                s.position_config()
            yaw_config = s.yaw_config()
            if yaw_config["yaw_rate_limit"] < .5:
                raise ValueError("This fixed schedule requires yaw_rate_limit >= 0.5 rad/s")
            s.speed_curriculum_frozen = True
            r = ParkingRecorder(raw) if position_enabled else YawRecorder(raw)
            if position_enabled:
                r.position_source = args.position_source
                if args.position_source == 'wheel-odometry':
                    s._position_measurement = lambda: (r.odom_xy,
                        raw.scene['robot'].data.root_link_lin_vel_b[:,0] if args.truth_lock_speed else r.wheel_vx)
            raw._stand_v2_acceptance = r
            n, dt = raw.num_envs, raw.physics_dt
            assert dt == .002 and raw.step_dt == .01 and raw.max_episode_length == 4000
            groups = torch.tensor([GROUPS[i % len(GROUPS)] for i in range(n)], device=raw.device)
            s.tracking_episode.fill_(True)
            s.zero_hold.fill_(False)
            s.tracking_acceleration.fill_(1.)
            s.tracking_velocity.zero_(); s.tracking_yaw_rate.zero_()
            s.tracking_ready_steps.zero_(); s.tracking_interval_steps.fill_(1000000)
            s.segment_started.zero_(); s.heading_hold.zero_(); s.yaw_active.zero_()
            schedule_tick = 0
            original_update = s._update_command

            def command(env_ids):
                t = schedule_tick*raw.step_dt
                vx_scale, yaw_scale = next(((v, y) for a,b,_,v,y in SEGMENTS if a <= t < b), (0., 0.))
                r.requests[:, 0] = groups[:, 0]*vx_scale
                r.requests[:, 1] = groups[:, 1]*yaw_scale
                s.tracking_episode.copy_(r.live)
                s.zero_hold.copy_(~r.live)
                s.tracking_velocity.copy_(torch.where(r.live, r.requests[:, 0], 0.))
                s.tracking_yaw_rate.copy_(torch.where(r.live, r.requests[:, 1], 0.))
                s.tracking_interval_steps.fill_(1000000)
                s.tracking_acceleration.fill_(1.)
                s.segment_started.zero_()
                original_update(env_ids)
                if position_enabled:
                    r.after_command()

            s._update_command = command
            raw.observation_manager.reset()
            raw.obs_buf = raw.observation_manager.compute(update_history=True)
            obs, policy = env.get_observations(), runner.get_inference_policy(device=args.device)
            action, observation = raw.action_manager.get_term("pd"), get_state(raw)
            initial = {k: v.cpu().numpy().copy() for k, v in dict(qpos=raw.sim.data.qpos,
                qvel=raw.sim.data.qvel, mass=raw.sim.model.body_mass, friction=raw.sim.model.geom_friction,
                inertia=raw.sim.model.body_inertia, com=raw.sim.model.body_ipos, solref=raw.sim.model.geom_solref,
                kp=action.kp_scale, kd=action.kd_scale, strength=action.motor_strength, nominal=action._nominal_pos,
                action_delay=action.action_delay_steps, observation_delay=observation.delay.lags).items()}
            np.savez_compressed(args.output/"initial.npz", **initial)
            stood = torch.zeros_like(r.live)
            first_end = torch.full((n,), float("nan"), device=raw.device)
            gate_violations = vx_slope_violations = yaw_slope_violations = 0
            resets = []
            for tick in range(4000):
                before = s.command[:, :2].clone()
                # command_manager在step末更新；到达8/14/...秒时立即切换下一拍命令。
                schedule_tick = tick+1
                obs, _, done, _ = env.step(policy(obs))
                stood |= r.live & s.has_stood
                ready = s.has_stood & ~s.recover & ~s.fallen & (s.tracking_ready_steps >= round(.5/raw.step_dt))
                gate_violations += int((r.live & (((s.command[:, 0] != 0.) & ~(ready & s.tracking_episode))
                    | ((s.command[:, 1] != 0.) & ~ready))).sum())
                vx_slope_violations += int((r.live & s.tracking_active &
                    ((s.command[:, 0]-before[:, 0]).abs() > raw.step_dt+2e-7)).sum())
                yaw_slope_violations += int((r.live & s.yaw_active &
                    ((s.command[:, 1]-before[:, 1]).abs() > yaw_config["yaw_acceleration"]*raw.step_dt+2e-7)).sum())
                failed = r.live & done.bool() & (s._isaac_failure | (tick < 3999))
                first_end[failed] = (tick+1)*raw.step_dt
                r.live &= ~failed
                if done.any() and tick < 3999:
                    ids = done.nonzero(as_tuple=False).flatten()
                    resets.append(dict(step=tick+1, env_ids=ids.cpu().tolist()))
                    raw.reset(env_ids=ids)
                    if position_enabled:
                        r.odom_xy[ids] = raw.sim.data.qpos[:, r.root_qadr][ids, :2]
                        r.wheel_vx[ids] = 0.
                        r.was_locked[ids] = False
                    obs = env.get_observations()
                if (tick+1) % 1000 == 0:
                    print(f"Yaw measurement {(tick+1)/100:.0f}/40s", flush=True)
            assert r.count == 20000
            assert frozen == tensor_digest(runner.alg.actor.state_dict())
            assert gate_violations == vx_slope_violations == yaw_slope_violations == 0
            physical, commands = r.trace.cpu().numpy(), r.command_trace.cpu().numpy()
            weights = initial["mass"].sum(-1)*np.linalg.norm(raw.sim.mj_model.opt.gravity)
            rollout = dict(episodes=[dict(ever_isaac_stood=bool(stood[i]), survived_to_end=bool(r.live[i])) for i in range(n)])
            balance = summarize(physical, weights, rollout, dt)
            rows = []
            for i in range(n):
                vx, yaw = GROUPS[i % len(GROUPS)]
                ended = float(first_end[i]) if torch.isfinite(first_end[i]) else None
                segments = []
                for start, end, kind, vx_scale, yaw_scale in SEGMENTS:
                    sl = slice(round(start/dt), round(end/dt))
                    previous_yaw = yaw if start == 14. else -yaw if start == 26. else 0.
                    segments.append(segment_metrics(physical[sl, i], commands[sl, i], start=start, end=end,
                        kind=kind, vx=vx*vx_scale, yaw=yaw*yaw_scale, previous_yaw=previous_yaw,
                        dt=dt, episode_end_s=ended, parking=position_enabled and vx*vx_scale == 0.))
                    if position_enabled and vx*vx_scale == 0.:
                        segments[-1]['parking'] = position_metrics(physical[sl, i], commands[sl, i],
                            r.position_trace[sl, i].cpu().numpy(), dt, s.cfg.position_tolerance,
                            terminated=ended is not None and ended <= end)
                b = balance[i]
                rows.append(dict(env=i, group_vx_m_s=vx, group_yaw_rad_s=yaw, ever_stood=bool(stood[i]),
                    first_end_s=ended,
                    survived_to_end=bool(r.live[i]), final1_balance_v31=bool(b["passed"]),
                    final_balance_hold_s=b["final_balance_hold_s"], balance_failure_reasons=b["failure_reasons"],
                    segments=segments))
                if position_enabled:
                    valid_locks = ((r.position_trace[:,i,0]>.5) & (r.trace[:,i,13]>.5)).nonzero(as_tuple=False).flatten()
                    rows[-1]['first_position_lock_s'] = float((valid_locks[0]+1)*dt) if len(valid_locks) else None
            np.savez_compressed(args.output/"trajectories.npz", physical=physical, command=commands,
                physical_fields=np.asarray(FIELDS), command_fields=np.asarray(COMMAND_FIELDS), weight_N=weights,
                group_requests=groups.cpu().numpy(), first_end_s=first_end.cpu().numpy(),
                time_s=(np.arange(r.count)+1)*dt,
                **(dict(position=r.position_trace.cpu().numpy(), position_fields=np.asarray(POSITION_FIELDS)) if position_enabled else {}))
            root = Path(__file__).resolve().parents[3]
            sources = [Path("scripts/agent/motion/assess.py"), Path(__file__), root/"scripts/agent/stand_v2/assess_chuanliantui_stand_v2.py",
                       *sorted((root/"mjlab_training/src/ct_mjlab").rglob("*.py"))]
            report = dict(motion_standard=STANDARD, motion_thresholds=MOTION_THRESHOLDS.copy(), contract="stand-v2-yaw-heading-measurement-v1", checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=digest(args.checkpoint), original_command_contract=contract,
                native_yaw_checkpoint=native, explicit_untrained_yaw_probe=not native,
                parameter_updates=0, actor_and_encoder_unchanged=True, actor_encoder_sha256=frozen,
                conditions=args.conditions, seed=args.seed, num_envs=n, duration_s=40., physics_dt_s=dt,
                policy_dt_s=raw.step_dt, yaw_config=yaw_config, vx_acceleration_m_s2=1.,
                schedule=SEGMENTS, groups=GROUPS, thresholds=THRESHOLDS,
                gate_violations=gate_violations, vx_slope_violations=vx_slope_violations,
                yaw_slope_violations=yaw_slope_violations, diagnostic_resets_after_first_result_frozen=resets,
                physical_fields=FIELDS, command_fields=COMMAND_FIELDS,
                balance_v31_final1=dict(total=n, passed=sum(row["final1_balance_v31"] for row in rows)),
                indicators=indicator_counts(rows), by_group={f"vx={vx},yaw={yaw}": indicator_counts([
                    row for row in rows if row["group_vx_m_s"] == vx and row["group_yaw_rad_s"] == yaw]) for vx,yaw in GROUPS},
                episodes=rows, xml_sha256=digest(MODEL_XML),
                sources={str(p.relative_to(root)): digest(args.output/'source.py') if p == Path(__file__) else digest(p) for p in sources},
                initial_sha256=digest(args.output/"initial.npz"), trajectories_sha256=digest(args.output/"trajectories.npz"),
                scope="Fixed first-episode schedule through production gating/slew; nonzero-rate statistics wait until both commands "
                      "reach requests, then settle 0.5s and require at least 0.5s measured data. Zero-yaw groups use heading hold. "
                      "Heading indicators require full-segment heading hold/gating and final-1s angular error <= 5deg. "
                      "Yaw/drift indicators are separate from unchanged balance v3.1, not a combined behavior approval. "
                      "A non-yaw checkpoint is an explicit untrained capability probe. No CPU/H7/hardware claim.")
            if position_enabled:
                report['position_hold'] = dict(config=s.position_config(), source=args.position_source,
                    lock_speed_source='truth' if args.position_source == 'truth' or args.truth_lock_speed else 'wheel-odometry',
                    native_checkpoint=native_position, explicit_untrained_probe=not native_position,
                    wheel_geometry=r.geometry, odometry_frequency_hz=1./raw.physics_dt,
                    odometry_scope='Signed mean wheel-joint speed times collision-mesh radius, integrated along ideal heading. '
                        'No slip or leg/pitch-motion compensation; truth anchor recorded at the same lock event. '
                        'Wheel-source feeds this odometry to parking; lock_speed_source records whether wheel velocity or diagnostic truth velocity arms the anchor. '
                        'Truth errors are recorded separately. '
                        'Current-heading error zero crossings during rotation can also arise from the changing projection; oscillation_candidate is diagnostic only.',
                    locked_envs=int((r.position_trace[:,:,0]>.5).any(0).sum()),
                    parking_segments=sum('parking' in segment for row in rows for segment in row['segments']))
            (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps({k:v for k,v in report.items() if k not in ("episodes", "sources", "by_group",
                "diagnostic_resets_after_first_result_frozen")}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--conditions", choices=("nominal", "training"), default="nominal")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num-envs", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-yaw", action="store_true", help="Explicit frozen yaw probe for a checkpoint trained without yaw")
    parser.add_argument('--probe-position-hold', action='store_true', help='Explicit untrained parking probe; requires yaw')
    parser.add_argument('--position-source', choices=('truth','wheel-odometry'), default='truth')
    parser.add_argument('--truth-lock-speed', action='store_true', help='Wheel-position diagnostic with truth lock velocity; isolates odometry drift from failure to arm')
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        if not args.checkpoint or not args.output or args.num_envs < 15 or args.num_envs % 15:
            parser.error("Need checkpoint/output and num-envs a positive multiple of 15")
        main(args)
