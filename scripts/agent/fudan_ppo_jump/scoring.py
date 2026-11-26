"""从连续 1 ms GPU 样本判定一次主动越条，测试夹具也走同一评分函数。"""
import numpy as np
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import BAR_X, BAR_DEPTH, BAR_HALF_WIDTH, TRIGGER, JUMP_TICKS
from scripts.agent.motion.assess import metrics_verdict
from .handoff import expected_landing_mode


def score(trace, modes, distances, height, *, landing_handoff=False, buffer_ticks=None):
    # 每行依次为左右轮心、轮底、双轮支撑、非轮接触力、重力、位置、世界速度、
    # 碰条、机体前向速度和偏航角速度。切模记录按每 10 ms 的实际策略步采集。
    result = dict(passed=False)
    def fail(reason):
        return dict(result, reason=reason)
    if (trace.ndim != 2 or trace.shape != (len(modes)*10, 23) or not len(trace)
            or distances.shape != modes.shape or not np.isfinite(trace).all()
            or not np.isfinite(distances).all()):
        return fail('missing_or_nonfinite')
    triggers = np.flatnonzero((distances < TRIGGER) & (distances > 0))
    if not len(triggers):
        return fail('no_trigger')
    tick = int(triggers[0])
    start = tick*10
    times = np.arange(len(modes))
    touchdown = -1
    if landing_handoff:
        expected,touchdown = expected_landing_mode(trace,tick,buffer_ticks)
    else:
        expected = (times>=tick) & (times<tick+JUMP_TICKS)
    if not np.array_equal(modes,expected):
        return fail('wrong_policy_window')
    x, y = trace[:, [0, 3]], trace[:, [1, 4]]
    bottom, contact = trace[:, 6:8], trace[:, 8:10] > .5
    result.update(trigger_distance_m=float(distances[tick]), approach_speed_m_s=float(trace[start, 21]),
                  touched_bar=bool(trace[:, 20].any()), peak_lower_wheel_m=float(bottom.min(1).max()))
    if trace[:, 20].any():
        return fail('hit_bar')
    if not contact[:start].all(1).any():
        return fail('no_ground_approach')
    near = np.flatnonzero((x >= BAR_X).any(1))
    if not len(near):
        return fail('no_crossing')
    enter = int(near[0])
    beyond = np.flatnonzero((x > BAR_X+BAR_DEPTH).all(1))
    if not len(beyond):
        return fail('incomplete_crossing')
    exit_tick = int(beyond[0])
    if enter < start or (np.abs(y[enter:exit_tick+1]) >= BAR_HALF_WIDTH).any():
        return fail('bypassed_bar')
    ground = np.flatnonzero(contact[:enter+1].any(1))
    if not len(ground) or contact[enter:exit_tick+1].any():
        has_flight = (~contact[start:exit_tick+1].any(1)).any()
        return fail('early_landing' if has_flight else 'no_takeoff')
    lift = int(ground[-1])+1
    if not start <= lift < len(trace) or not modes[lift//10] or not (trace[max(start,lift-10):lift+1, 19] > 0).any():
        return fail('no_commanded_takeoff')
    over = (x >= BAR_X) & (x <= BAR_X+BAR_DEPTH)
    if not over.any(0).all():
        return fail('missing_edge_samples')
    result['wheel_clearance_over_bar_m'] = float((bottom[over]-height).min())
    if result['wheel_clearance_over_bar_m'] <= 0:
        return fail('insufficient_clearance')
    landed = np.flatnonzero(contact[exit_tick:].all(1))
    if not len(landed):
        return fail('no_far_landing')
    land = exit_tick + int(landed[0])
    if landing_handoff and touchdown<0:
        return fail('no_confirmed_landing_handoff')
    if landing_handoff:
        result['touchdown_policy_tick'] = touchdown
        result['jump_policy_steps'] = int(modes.sum())
    end = land + 1000
    if end > len(trace):
        return fail('missing_landing_window')
    if (not (x[land:end] > BAR_X+BAR_DEPTH).all() or
            (trace[start:end, 10] > 1.).any() or (trace[start:end, 13] > -.1).any()):
        return fail('unstable_landing')
    # 一秒窗口内须持续向前并保持轮支撑，排除二次弹跳。
    window = trace[land:end]
    if (window[:, 21] <= 0).any() or not contact[land:end].any(1).all():
        return fail('lost_walking_after_landing')
    result['airtime_s'] = (np.flatnonzero(contact[lift:].any(1))[0]).item()*.001
    window = window[500:]  # 预声明的后半秒用作稳定巡航段。
    result['post_landing_motion'] = metrics_verdict(vx_mae=float(np.abs(window[:,21]-2.).mean()),
        yaw_mae=float(np.abs(window[:,22]).mean()),
        pitch_peak=float(np.abs(np.degrees(np.arcsin(window[:,11].clip(-1,1)))).max()), complete=True)
    return dict(result, passed=True, reason='cleared_and_walked')


def check_scoring():
    n = 3000
    x = BAR_X - 2.6 + np.arange(n)*.002
    distance = BAR_X-x[::10]
    switch = np.flatnonzero((distance < TRIGGER) & (distance > 0))[0]
    modes = (np.arange(n//10) >= switch) & (np.arange(n//10) < switch + JUMP_TICKS)
    trace = np.zeros((n,23))
    trace[:,0] = trace[:,3] = trace[:,14] = x
    trace[:,8:10] = 1
    trace[:,13] = -1.
    trace[:,17] = trace[:,21] = 2.
    trace[1190:1410,8:10] = 0
    trace[1180:1400,19] = .4
    trace[1190:1410,6:8] = .14
    assert score(trace,modes,distance,.10)['passed']
    landing_modes, touchdown = expected_landing_mode(trace,switch)
    assert not landing_modes[:switch].any() and landing_modes[switch]
    assert touchdown == 141 and landing_modes[161] and not landing_modes[162]
    assert score(trace,landing_modes,distance,.10,landing_handoff=True)['passed']
    assert score(trace,modes,distance,.10,landing_handoff=True)['reason']=='wrong_policy_window'
    reasons = {}
    for name, rows, cols, value in [('no_jump', slice(None), slice(8,10), 1),
            ('collision', 1310, 20, 1), ('bypass', slice(None), 1, 2.),
            ('fall', 1600, 13, .3), ('early_landing', 1320, 8, 1)]:
        case = trace.copy()
        case[rows,cols] = value
        verdict = score(case,modes,distance,.10)
        assert not verdict['passed'], name
        reasons[name] = verdict['reason']
    return reasons
