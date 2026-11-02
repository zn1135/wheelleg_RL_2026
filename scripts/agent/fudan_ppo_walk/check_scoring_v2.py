"""Formal synthetic checks for acceptance v2; no simulator or policy imports."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np

from ct_mjlab.fudan_ppo_walk.protocol import samples
from .check_scoring import fixture
from .score import score as score_v1
from .score_v2 import acceptance_manifest, load_window, score, summarize_recording, tick


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sample = next(s for s in samples() if s['case']['id'] == 'forward_2_0_0.26')
    clean = fixture(sample)
    checks = []

    def verify(name, condition):
        assert condition, name
        checks.append(name)

    verify('clean_tracking_and_load', score(sample, clean)['passed'])
    for name,sid in [('spin', 'spin_0_2_0.26'), ('stop', 'stop_1_0.26'), ('variable_height', 'height_1')]:
        case = next(s for s in samples() if s['case']['id'] == sid)
        verify(name + '_clean', score(case, fixture(case))['passed'])
    bad = copy.deepcopy(clean)
    bad['wheel_support'][-20,0] = 0.
    verify('single_wheel_point_v1_fail_v2_pass', not score_v1(sample,bad)['passed'] and score(sample,bad)['passed'])
    for length in (9,10):
        bad = copy.deepcopy(clean)
        bad['wheel_support'][1500:1500+length] = 0.
        result = score(sample,bad)
        verify(f'raw_simultaneous_low_{length*2}ms', result['passed'] == (length == 9))
        verify(f'raw_low_not_hidden_by_mean_{length*2}ms', result['load_windows'][0]['fraction_passed'])
    bad = copy.deepcopy(clean)
    bad['wheel_support'][100:200] = 0.
    verify('root_reset_low_support_outside_measurement', score(sample,bad)['passed'])
    for length in (50,51):
        bad = copy.deepcopy(clean)
        bad['wheel_support'][1500:1500+length] = 30.
        result = score(sample,bad)
        window = result['load_windows'][0]
        # 手工结果：连续L点每轮30N，20ms平均越界的窗口恰有L-1个。
        verify(f'bad_mean_run_{(length-1)*2}ms', window['bad_window_runs']['maximum_run_ticks'] == length-1)
        verify(f'100ms_boundary_{length}', result['passed'] == (length == 50) and window['fraction_passed'])
    for length in (11,12):
        small = dict(wheel_support=np.full((209,2),50.),weight=np.full(209,100.))
        small['wheel_support'][80:80+length] = 30.
        window = load_window(small,0,209,'fraction_boundary')
        verify(f'95percent_boundary_{length}', window['valid_windows'] == 200 and window['joint_valid_windows'] == 201-length)
        verify(f'95percent_inclusive_{length}', window['passed'] == (length == 11))

    # 独立逐窗口求和与连续分组，核对向量算法的窗口对齐和计数。
    rng = np.random.default_rng(1202)
    random_trace = dict(wheel_support=rng.uniform(0.,120.,(211,2)),weight=rng.uniform(90.,110.,211))
    expected = []
    for i in range(17,194-10+1):
        wheels = np.sum(random_trace['wheel_support'][i:i+10],axis=0) / 10
        weight = np.sum(random_trace['weight'][i:i+10]) / 10
        expected.append(bool(min(wheels) >= max(.1*weight,5.) and .8*weight <= sum(wheels) <= 1.2*weight))
    window = load_window(random_trace,17,194,'independent_oracle')
    longest = max((len(list(g)) for v,g in itertools.groupby(expected) if not v), default=0)
    verify('independent_window_oracle', window['joint_valid_windows'] == sum(expected) and window['valid_windows'] == len(expected)
        and window['bad_window_runs']['maximum_run_ticks'] == longest)
    verify('no_partial_mean_window', not load_window(clean,1000,1009,'too_short')['passed'])

    for length in (10,11):
        bad = copy.deepcopy(clean)
        bad['base_contact'][100:100+length] = 50.
        result = score(sample,bad)
        verify(f'startup_contact_{length*2}ms', result['passed'] == (length == 10))
        verify(f'startup_peak_still_retained_{length}', result['body_contact']['raw_peak_N'] == 50. and result['v1_diagnostic']['fell'])
    for total in (25,26):
        bad = copy.deepcopy(clean)
        for lo,length in [(100,10),(200,10),(300,total-20)]:bad['base_contact'][lo:lo+length] = 50.
        result = score(sample,bad)
        verify(f'startup_total_{total*2}ms', result['passed'] == (total == 25)
            and result['body_contact']['startup_over10N']['maximum_run_ticks'] == 10)
    bad = copy.deepcopy(clean)
    bad['base_contact'][999] = 389.
    verify('pre2s_contact_not_in_post2s_mean', score(sample,bad)['passed'])
    for peak,passed in [(10.,True),(10.01,False)]:
        bad = copy.deepcopy(clean)
        bad['base_contact'][1000] = peak
        result = score(sample,bad)
        verify(f'post2s_single_point_{peak}N', result['passed'] == passed)
    bad = copy.deepcopy(clean)
    bad['base_contact'][-2] = 10.01
    verify('last_force_interval_is_checked', not score(sample,bad)['passed'])
    bad = copy.deepcopy(clean)
    bad['base_contact'][-1] = 389.
    result = score(sample,bad)
    verify('terminal_state_not_a_force_duration', result['passed'] and result['body_contact']['raw_peak_N'] == 389.)
    for value in (1.,1.01):
        bad = copy.deepcopy(clean)
        bad['base_contact'][1000:-1] = value
        verify(f'post2s_mean_body_{value}N', score(sample,bad)['passed'] == (value == 1.))
    bad = copy.deepcopy(clean)
    bad['base_contact'][100:500] = 50.
    verify('sustained_startup_body_support_rejected', not score(sample,bad)['passed'])
    height_case = next(s for s in samples() if s['case']['id'] == 'height_1')
    bad = fixture(height_case)
    bad['base_contact'][tick(7.1):tick(7.12)] = 50.
    verify('command_change_does_not_restart_contact_grace', not score(height_case,bad)['passed'])
    bad = copy.deepcopy(clean)
    bad['wheel_support'][1200:1800] = 0.
    verify('sustained_raw_support_loss_rejected', not score(sample,bad)['passed'])

    for field in ('linear_body','angular_body','position'):
        bad = copy.deepcopy(clean)
        component = 2 if field == 'angular_body' else (2 if field == 'position' else 0)
        bad[field][1000:3500,component] += 1.
        verify('original_tracking_gate_' + field, not score(sample,bad)['passed'])
    for angle,passed in [(10.,True),(10.01,False)]:
        bad = copy.deepcopy(clean)
        bad['roll_pitch_deg'][1500,1] = angle
        verify(f'measurement_pitch_{angle}', score(sample,bad)['passed'] == passed)
    for angle,passed in [(60.,True),(60.01,False)]:
        bad = copy.deepcopy(clean)
        bad['roll_pitch_deg'][100,0] = angle
        verify(f'whole_trajectory_tilt_{angle}', score(sample,bad)['passed'] == passed)
    for field,component,values in [('roll_pitch_deg',0,(35.,35.01)),('angular_body',0,(.5,.501))]:
        for i,value in enumerate(values):
            bad = copy.deepcopy(clean)
            bad[field][-20,component] = value
            verify(f'final_second_{field}_{value}', score(sample,bad)['passed'] == (i == 0))
    stop_case = next(s for s in samples() if s['case']['id'] == 'stop_1_0.26')
    bad = fixture(stop_case)
    bad['linear_world'][5000,0] = .101
    verify('original_stop_speed_gate', not score(stop_case,bad)['passed'])
    bad = fixture(stop_case)
    bad['wheel_support'][5000:5010] = 0.
    verify('stop_observation_raw_support_gate', not score(stop_case,bad)['passed'])
    for field in ('terminated','numerical_failure'):
        bad = copy.deepcopy(clean)
        bad[field][1] = True
        verify(field + '_rejected', not score(sample,bad)['passed'])
    for field in clean:
        bad = {k:v for k,v in clean.items() if k != field}
        verify('missing_' + field, not score(sample,bad)['complete'] and not score(sample,bad)['passed'])
    bad = copy.deepcopy(clean)
    bad['linear_body'][100,0] = np.nan
    verify('nan_rejected', score(sample,bad)['numerical_failure'] and not score(sample,bad)['passed'])
    bad = copy.deepcopy(clean)
    bad['weight'][100] = 0.
    verify('invalid_weight_rejected', not score(sample,bad)['passed'])
    bad = {k:v[:-1] for k,v in clean.items()}
    verify('truncated_trace_rejected', not score(sample,bad)['complete'])
    bad['numerical_failure']=bad['numerical_failure'].copy()
    bad['numerical_failure'][-1]=True
    verify('truncated_trace_retains_numerical_failure',score(sample,bad)['numerical_failure'] and not score(sample,bad)['passed'])
    bad = copy.deepcopy(clean)
    bad['time'][1000] += .001
    verify('wrong_time_grid_rejected', not score(sample,bad)['passed'])
    bad = copy.deepcopy(clean)
    bad['time'][1000] -= 1e-8
    result = score(sample,bad)
    verify('accepted_time_precision_still_uses_integer_ticks', result['passed'] and result['load_windows'][0]['start_tick'] == 1000)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory/'traces').mkdir()
        missing = dict(sample,id='missing_diagnostic')
        (directory/'protocol.json').write_text(json.dumps(dict(mode='diagnostic_not_formal_acceptance',samples=[sample,missing])))
        np.savez(directory/'traces'/(sample['id']+'.npz'),**clean)
        report = summarize_recording(directory)
        verify('missing_trace_keeps_denominator', report['total'] == 2 and report['complete'] == report['technical_passed'] == 1)
        verify('diagnostics_never_formal_acceptance', report['formal_matrix_thresholds_met'] is None and report['final_acceptance'] == 'pending_user_video_review')
        (directory/'protocol.json').write_text(json.dumps(dict(mode='formal_987_samples',samples=[sample])))
        try:summarize_recording(directory)
        except ValueError:rejected = True
        else:rejected = False
        verify('shortened_formal_matrix_rejected', rejected)
    result = dict(passed=True,protocol=acceptance_manifest(),checks=checks,count=len(checks),
        physics_steps=0,model_forwards=0,optimizer_updates=0,
        scope='Synthetic trajectories and independent window arithmetic, not robot behavior acceptance')
    with args.output.open('x') as output:output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),physics_steps=0,model_forwards=0,optimizer_updates=0)))


if __name__ == '__main__':
    main()
