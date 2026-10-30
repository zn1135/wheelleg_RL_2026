#!/usr/bin/env python3
"""Locked complete-run data and per-segment scoring for encoder-only experiments.

This module never optimizes a network or advances physics. Predictions passed to
score_predictions are m/s; sampled supervision retains the original COM*2 scale.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_walk_velocity_compensation as phase_source


ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA = 'c409bb9a4c28308c5919541d2bb4c5082f43fd9d904e40c8b5fcfc1a33dd3708'
BUCKETS = ('initial','parking','cruise_forward','cruise_backward','start','brake')
AXES = ('x','y','z')
METRICS = ('mae_m_s','p95_absolute_error_m_s','bias_m_s','abs_bias_m_s',
           'peak_absolute_error_m_s','peak_time_s','band_1_3Hz_rms_m_s')
OFFLINE_GATES = {
    'offline_vx_gates': {
        'initial': {'mae_increase_max':.002,'p95_increase_max':.005,'abs_bias_increase_max':.002,'peak_increase_max':.005},
        'parking': {'mae_ratio_max':.8,'abs_bias_ratio_max':.8,'p95_increase_max':0.,'band_1_3Hz_rms_ratio_max':1.,'peak_increase_max':.005},
        'cruise_forward_start_brake': {'mae_increase_max':.002,'p95_increase_max':.005,'abs_bias_increase_max':.002,'peak_increase_max':.02},
        'cruise_backward': {'mae_ratio_max':.8,'abs_bias_ratio_max':.8,'p95_increase_max':0.,'peak_increase_max':.02},
    },
    'offline_other_axes_each_phase': {'mae_increase_max':.002,'p95_increase_max':.005,'abs_bias_increase_max':.002,'peak_increase_max':.005},
    'small_bias_and_parking_band_exception': {'baseline_below':.001,'absolute_increase_max':.0005},
}


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):
            result.update(part)
    return result.hexdigest()


def stable_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(str(value.dtype).encode())
    result.update(np.asarray(value.shape,dtype=np.int64).tobytes())
    result.update(value.tobytes())
    return result.hexdigest()


def validate_gate_card(card):
    if not isinstance(card,dict):
        card = json.loads(Path(card).read_text())
    assert {key:card[key] for key in OFFLINE_GATES} == OFFLINE_GATES, 'Offline gate/card mismatch'
    return stable_sha(OFFLINE_GATES)


def rebuild_history(frames):
    indices = np.maximum(0,np.arange(len(frames))[:,None]+np.arange(-4,1))
    return frames[indices].transpose(0,2,1,3).reshape(len(frames),frames.shape[1],125)


def _segments(case, seconds):
    if 'segments' not in case:
        raise ValueError('Training/scoring needs explicit preregistered command segments')
    output = []
    for item in case['segments']:
        row = dict(start=float(item['start']),end=float(item['end']),speed=float(item['speed']),
            acceleration=float(item.get('acceleration',case['acceleration'])))
        assert row['end'] > row['start'] and row['acceleration'] > 0.
        if output:
            assert output[-1]['end'] == row['start']
        if output and output[-1]['speed'] == row['speed']:
            output[-1]['end'] = row['end']
        else:
            output.append(row)
    assert output[0]['start'] == 0. and output[0]['speed'] == 0. and output[-1]['end'] == seconds
    return output


@dataclass
class PhaseData:
    env: int
    bucket: str
    segment: int | None
    start: float | None
    end: float | None
    indices: np.ndarray
    expected_samples: int
    complete: bool
    status: str
    spec: dict


class RunData:
    def __init__(self,run_id,folder,expected_files=None):
        self.id,self.folder = str(run_id),Path(folder).resolve()
        metadata_path, trace_path = self.folder/'encoder.json',self.folder/'encoder.npz'
        self.meta = json.loads(metadata_path.read_text())
        meta = self.meta
        assert meta['checkpoint_sha256'] == PARENT_SHA
        assert meta['parameter_updates'] == 0 and meta['actor_unchanged'] and meta['encoder_unchanged']
        assert meta['feedback_mode'] == meta['velocity_source'] == 'encoder'
        assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']
        assert meta['policy_dt'] == .01 and meta['dt'] == .002
        receipt_path = self.folder/'result.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['verified']
        self.files = {str(path):file_sha(path) for path in (metadata_path,trace_path,receipt_path,self.folder/'source.py')}
        if expected_files is not None:
            assert self.files == expected_files, 'Data differs from inventoried/sealed complete run'
        checkpoint = Path(meta['checkpoint'])
        assert file_sha(checkpoint) == PARENT_SHA
        self.files[str(checkpoint)] = PARENT_SHA
        with np.load(trace_path,allow_pickle=False) as data:
            self.observation = data['policy_observation']
            self.target_m_s = data['target_com_m_s']
            self.valid = data['policy_valid']
            self.times = data['policy_time_s']
            self.original_m_s = data['encoder_m_s']
            self.qpos_live = data['qpos_live']
            self.reset_after = data['reset_after_policy'] if 'reset_after_policy' in data.files else None
            native_history = data['native_history'] if 'native_history' in data.files else None
        ticks, envs = self.valid.shape
        assert self.observation.shape == (ticks,envs,25) and self.observation.dtype == np.float32
        assert self.target_m_s.shape == self.original_m_s.shape == (ticks,envs,3)
        assert self.valid.dtype == np.bool_ and self.valid[0].all()
        assert not (self.valid[1:] & ~self.valid[:-1]).any(), 'Do not concatenate reset episodes'
        assert np.array_equal(self.times,np.arange(ticks)*.01)
        assert np.isfinite(self.observation[self.valid]).all() and np.isfinite(self.target_m_s[self.valid]).all()
        assert np.array_equal(self.qpos_live[-1],~np.asarray(meta['episode_done'],dtype=bool))
        assert int(self.qpos_live[-1].sum()) == meta['survived']
        assert np.array_equal(self.valid.sum(0),np.rint(np.asarray(meta['episode_end'])/.01).astype(int))
        history = rebuild_history(self.observation)
        if native_history is not None:
            assert native_history.shape == history.shape
            assert np.array_equal(history[self.valid],native_history[self.valid]), 'Native history does not match fixed old-to-new rule'
            self.history = native_history
            self.history_source = 'native125; byte-equal to source-verified five-frame reconstruction on first-episode valid samples'
        else:
            self.history = history
            self.history_source = 'deterministic exact25 first-frame backfill/old-to-new; prior parent-output parity required by data card'
        self.target_scaled = self.target_m_s*np.float32(2.)
        self.phases = []
        self.applicability = {}
        covered = []
        for case in meta['cases']:
            segments = _segments(case,meta['seconds'])
            expected = {'initial','parking'}
            for previous,current in zip(segments,segments[1:]):
                expected.add('brake' if current['speed'] == 0. else 'start')
                expected.add('parking' if current['speed'] == 0. else 'cruise_forward' if current['speed'] > 0. else 'cruise_backward')
            for env in case['env_ids']:
                covered.append(env)
                self.applicability[env] = sorted(expected)
                for spec in phase_source.phase_specs(segments,self.times,self.observation[:,env,6]/2.,self.valid[:,env]):
                    bucket = spec['kind']
                    if bucket == 'cruise':
                        bucket += '_forward' if spec['requested_speed'] > 0. else '_backward'
                    mask = (self.times >= spec['start']-1e-9) & (self.times < spec['end']-1e-9)
                    indices = np.flatnonzero(mask & self.valid[:,env])
                    count = int(mask.sum())
                    complete = bool(count and len(indices) == count)
                    self.phases.append(PhaseData(env,bucket,spec['segment'],spec['start'],spec['end'],
                        indices,count,complete,'complete' if complete else 'missing',spec))
                for bucket in BUCKETS:
                    if bucket not in expected:
                        self.phases.append(PhaseData(env,bucket,None,None,None,np.empty(0,dtype=np.int64),0,
                            False,'not_applicable',{'reason':'No such requested-speed phase in this condition'}))
        assert sorted(covered) == list(range(envs))
        self.sampling = {bucket:defaultdict(list) for bucket in BUCKETS}
        for phase in self.phases:
            if phase.complete:
                self.sampling[phase.bucket][phase.env].append(phase.indices)
        self.envs = envs

    def lock(self):
        return dict(id=self.id,folder=str(self.folder),seed=self.meta['seed'],files=self.files,
            parent_model_sha256=PARENT_SHA,history_source=self.history_source,
            history_sha256=array_sha(self.history),target_scaled_sha256=array_sha(self.target_scaled),
            valid_sha256=array_sha(self.valid),phase_index_sha256=stable_sha([
                dict(env=p.env,bucket=p.bucket,segment=p.segment,start=p.start,end=p.end,
                    expected=p.expected_samples,status=p.status,index_sha256=array_sha(p.indices)) for p in self.phases]),
            phase_counts=dict(Counter(p.bucket for p in self.phases if p.complete)),
            applicability=self.applicability)


class SplitData:
    def __init__(self,name,runs):
        self.name,self.runs = str(name),list(runs)
        assert self.runs and len({run.id for run in self.runs}) == len(self.runs)
        self._lock = dict(name=self.name,runs=[run.lock() for run in self.runs],
            bucket_order=BUCKETS,sampling='uniform bucket, then run, eligible episode, contiguous segment, then frame; with replacement',
            phase_source_path=str(Path(phase_source.__file__).resolve()),phase_source_sha256=file_sha(phase_source.__file__),
            library_source_path=str(Path(__file__).resolve()),library_source_sha256=file_sha(__file__),
            gate_sha256=stable_sha(OFFLINE_GATES))
        self.lock_sha = stable_sha(self._lock)
        self.available_runs = {bucket:[run for run in self.runs if run.sampling[bucket]] for bucket in BUCKETS}
        self.sampling_counts = dict(total=0,buckets={bucket:0 for bucket in BUCKETS},
            runs={run.id:0 for run in self.runs},
            bucket_runs={bucket:{run.id:0 for run in self.runs} for bucket in BUCKETS})

    def lock(self):
        return self._lock

    def verify_files(self):
        for run in self.runs:
            for path,expected in run.files.items():
                assert file_sha(path) == expected, path
        assert file_sha(phase_source.__file__) == self._lock['phase_source_sha256']
        assert file_sha(__file__) == self._lock['library_source_sha256']
        if hasattr(self,'data_card'):
            assert file_sha(self.data_card['path']) == self.data_card['sha256']
        return True

    def verify_parent_predictions(self,predictions,tolerance=1e-5):
        assert tolerance == 1e-5, 'Fixed parent-replay tolerance'
        output = {}
        assert set(predictions) == {run.id for run in self.runs}
        for run in self.runs:
            value = np.asarray(predictions[run.id])
            assert value.shape == run.original_m_s.shape and np.isfinite(value[run.valid]).all()
            maximum = float(abs(value[run.valid]-run.original_m_s[run.valid]).max())
            output[run.id] = dict(max_abs_m_s=maximum,tolerance_m_s=tolerance,passed=maximum <= tolerance)
        assert all(row['passed'] for row in output.values()), 'Parent encoder replay failure'
        return output


def load_datasets(data_card):
    card_path = Path(data_card).resolve()
    card = json.loads(card_path.read_text())
    assert card['parent_model_sha256'] == PARENT_SHA
    assert card['legacy_history_parity']['passed']
    specs = {row['id']:row for row in card['runs']}
    assignments = {'train':card['proposed_split']['primary_train_whole_runs'],
                   'dev':card['proposed_split']['primary_development_whole_runs']}
    assert assignments == {'train':['compensation44','shadow46'],'dev':['compensation45','shadow47']}
    assert not set(assignments['train']) & set(assignments['dev'])
    result = {}
    for name,ids in assignments.items():
        runs = [RunData(run_id,specs[run_id]['folder'],specs[run_id]['files_sha256']) for run_id in ids]
        assert all(run.valid.all() and not any(run.meta['episode_done']) for run in runs)
        split = SplitData(name,runs)
        assert all(len(split.available_runs[bucket]) == len(runs) for bucket in BUCKETS)
        split.data_card = dict(path=str(card_path),sha256=file_sha(card_path))
        split._lock['data_card'] = split.data_card
        split.lock_sha = stable_sha(split._lock)
        result[name] = split
    return result


def load_split(run_folders,name='test'):
    """Load future sealed test runs without admitting them to training splits."""
    return SplitData(name,[RunData(run_id,folder) for run_id,folder in run_folders.items()])


def sample_batch(split,rng,batch_size):
    assert split.name == 'train', 'No development/test sampling for optimizer updates'
    assert isinstance(rng,np.random.Generator) and batch_size > 0
    history = np.empty((batch_size,125),dtype=np.float32)
    target = np.empty((batch_size,3),dtype=np.float32)
    buckets = rng.integers(len(BUCKETS),size=batch_size)
    for sample,bucket_index in enumerate(buckets):
        bucket = BUCKETS[bucket_index]
        available = split.available_runs[bucket]
        assert available, 'Missing required training bucket'
        run = available[int(rng.integers(len(available)))]
        episodes = list(run.sampling[bucket])
        env = episodes[int(rng.integers(len(episodes)))]
        segments = run.sampling[bucket][env]
        indices = segments[int(rng.integers(len(segments)))]
        tick = int(indices[int(rng.integers(len(indices)))])
        history[sample],target[sample] = run.history[tick,env],run.target_scaled[tick,env]
        split.sampling_counts['total'] += 1
        split.sampling_counts['buckets'][bucket] += 1
        split.sampling_counts['runs'][run.id] += 1
        split.sampling_counts['bucket_runs'][bucket][run.id] += 1
    assert np.isfinite(history).all() and np.isfinite(target).all()
    return history,target


def _description(values):
    values = np.asarray(values,dtype=float)
    values = values[np.isfinite(values)]
    return dict(count=len(values),minimum=None if not len(values) else float(values.min()),
        median=None if not len(values) else float(np.median(values)),maximum=None if not len(values) else float(values.max()),
        mean=None if not len(values) else float(values.mean()))


def score_predictions(split,predictions,baseline=None):
    assert set(predictions) == {run.id for run in split.runs}
    rows,segments = [],[]
    failed_episodes,required_missing,long_parking = [],[],[]
    for run in split.runs:
        predicted = np.asarray(predictions[run.id])
        assert predicted.shape == run.target_m_s.shape, 'm/s predictions must match same-tick3-axis target'
        failed_episodes.extend(dict(run=run.id,env=env,end_s=run.meta['episode_end'][env])
            for env,failed in enumerate(run.meta['episode_done']) if failed)
        complete_long = set()
        for phase in run.phases:
            indices = phase.indices
            candidate_finite = bool(len(indices) and np.isfinite(predicted[indices,phase.env]).all())
            status = phase.status if phase.status != 'complete' else 'complete' if candidate_finite else 'missing'
            if status == 'missing':
                required_missing.append(dict(run=run.id,env=phase.env,bucket=phase.bucket,segment=phase.segment))
            metrics = np.full((3,len(METRICS)),np.nan)
            lags = [None]*3
            vector_peak = None
            if status == 'complete':
                # Same indices for prediction/target; correlation lag never changes scoring labels.
                actual = predicted[indices,phase.env]
                truth = run.target_m_s[indices,phase.env]
                error = actual-truth
                absolute = abs(error)
                maximum_indices = absolute.argmax(0)
                metrics[:,:6] = np.stack((absolute.mean(0),np.percentile(absolute,95,axis=0),error.mean(0),
                    abs(error.mean(0)),absolute.max(0),run.times[indices[maximum_indices]]),axis=1)
                norms = np.linalg.norm(error,axis=1)
                peak_index = int(norms.argmax())
                vector_peak = dict(value_m_s=float(norms[peak_index]),time_s=float(run.times[indices[peak_index]]),
                    xyz_error_m_s=error[peak_index].tolist())
                if phase.bucket == 'parking' and phase.expected_samples >= 300:
                    centered = error-error.mean(0,keepdims=True)
                    spectrum = np.fft.rfft(centered,axis=0)
                    frequencies = np.fft.rfftfreq(len(error),d=.01)
                    spectrum[(frequencies < 1.) | (frequencies > 3.)] = 0.
                    band = np.fft.irfft(spectrum,n=len(error),axis=0)
                    metrics[:,6] = np.sqrt((band**2).mean(0))
                    complete_long.add(phase.env)
                for axis in range(3):
                    lags[axis] = phase_source.lag_measurement(actual[:,axis],truth[:,axis])
                    lags[axis]['derived_truth_timing_uncertainty_ms'] = 0
            segments.append(dict(run=run.id,env=phase.env,bucket=phase.bucket,segment=phase.segment,
                status=status,vector_peak_error=vector_peak))
            for axis_index,axis in enumerate(AXES):
                values = None if status != 'complete' else {key:None if not np.isfinite(value) else float(value)
                    for key,value in zip(METRICS,metrics[axis_index])}
                rows.append(dict(run=run.id,env=phase.env,bucket=phase.bucket,segment=phase.segment,axis=axis,
                    start_s=phase.start,end_s=phase.end,status=status,expected_samples=phase.expected_samples,
                    valid_samples=len(indices),metrics=values,lag=lags[axis_index],
                    band_status='missing' if status == 'missing' and phase.bucket == 'parking' else
                        'available' if status == 'complete' and phase.bucket == 'parking' and phase.expected_samples >= 300 else 'not_applicable'))
        long_parking.extend(dict(run=run.id,env=env) for env in range(run.envs) if env not in complete_long)
    summary = {}
    for bucket in BUCKETS:
        summary[bucket] = {}
        for axis in AXES:
            selected = [row for row in rows if row['bucket'] == bucket and row['axis'] == axis]
            complete = [row for row in selected if row['status'] == 'complete']
            summary[bucket][axis] = {key:_description([row['metrics'][key] for row in complete if row['metrics'][key] is not None]) for key in METRICS}
            summary[bucket][axis]['coverage'] = dict(Counter(row['status'] for row in selected))
    result = dict(split=split.name,data_lock_sha256=split.lock_sha,parent_model_sha256=PARENT_SHA,
        predictions_sha256={key:array_sha(value) for key,value in predictions.items()},
        units='Predictions and metrics m/s; supervision target scale2 is removed before scoring',
        label_alignment='Same current-policy tick; no shifting or lag alignment',rows=rows,segments=segments,summary=summary,
        coverage=dict(failed_episodes=failed_episodes,required_missing_phases=required_missing,
            missing_complete_3s_parking_envs=long_parking,not_applicable_rows=sum(row['status'] == 'not_applicable' for row in rows)))
    if baseline is not None:
        result['gate'] = candidate_gate(result,baseline)
    return result


def _row_key(row):
    return row['run'],row['env'],row['bucket'],row['segment'],row['axis']


def candidate_gate(candidate,baseline,gate_card=None):
    if gate_card is not None:
        validate_gate_card(gate_card)
    assert candidate['data_lock_sha256'] == baseline['data_lock_sha256']
    original = {_row_key(row):row for row in baseline['rows']}
    assert len(original) == len(baseline['rows'])
    assert set(original) == {_row_key(row) for row in candidate['rows']}
    output,failures = [],Counter()
    for row in candidate['rows']:
        base = original[_row_key(row)]
        assert (row['start_s'],row['end_s'],row['expected_samples']) == (base['start_s'],base['end_s'],base['expected_samples'])
        if row['status'] == base['status'] == 'not_applicable':
            output.append(dict(key=_row_key(row),status='not_applicable',passed=None,checks={}))
            continue
        checks = {}
        if row['status'] != 'complete' or base['status'] != 'complete':
            checks['complete_expected_phase'] = False
        else:
            c,b = row['metrics'],base['metrics']
            if row['axis'] != 'x':
                rule = OFFLINE_GATES['offline_other_axes_each_phase']
            else:
                name = row['bucket'] if row['bucket'] in ('initial','parking','cruise_backward') else 'cruise_forward_start_brake'
                rule = OFFLINE_GATES['offline_vx_gates'][name]
            for field,metric in (('mae','mae_m_s'),('p95','p95_absolute_error_m_s'),('abs_bias','abs_bias_m_s'),('peak','peak_absolute_error_m_s')):
                if field+'_ratio_max' in rule:
                    maximum = b[metric]*rule[field+'_ratio_max']
                    exception = OFFLINE_GATES['small_bias_and_parking_band_exception']
                    if field == 'abs_bias' and b[metric] < exception['baseline_below']:
                        maximum = b[metric]+exception['absolute_increase_max']
                else:
                    maximum = b[metric]+rule[field+'_increase_max']
                checks[field] = bool(c[metric] <= maximum)
            if row['axis'] == 'x' and row['bucket'] == 'parking' and row['expected_samples'] >= 300:
                field = 'band_1_3Hz_rms_m_s'
                exception = OFFLINE_GATES['small_bias_and_parking_band_exception']
                if c[field] is None or b[field] is None:
                    checks['band_1_3Hz_rms'] = False
                else:
                    maximum = b[field]*rule['band_1_3Hz_rms_ratio_max']
                    if b[field] < exception['baseline_below']:
                        maximum = b[field]+exception['absolute_increase_max']
                    checks['band_1_3Hz_rms'] = bool(c[field] <= maximum)
        passed = all(checks.values())
        failures.update(key for key,value in checks.items() if not value)
        output.append(dict(key=_row_key(row),status='checked',passed=passed,checks=checks))
    coverage = {}
    for name in ('failed_episodes','required_missing_phases','missing_complete_3s_parking_envs'):
        coverage[name] = not candidate['coverage'][name] and not baseline['coverage'][name]
        if not coverage[name]:
            failures['coverage_'+name] += 1
    passed = all(coverage.values()) and all(row['passed'] is not False for row in output)
    return dict(passed=bool(passed),support=bool(passed),gate_sha256=stable_sha(OFFLINE_GATES),
        failure_counts=dict(failures),coverage_checks=coverage,rows=output,
        scope='Every applicable phase/environment/axis must pass; summary means never decide the gate')


def log_scalars(score):
    output = {}
    for bucket,axes in score['summary'].items():
        for axis,metrics in axes.items():
            for metric,description in metrics.items():
                if metric in ('coverage','peak_time_s'):
                    continue
                for statistic in ('mean','minimum','median','maximum'):
                    value = description[statistic]
                    if value is not None:
                        output[f'{bucket}/{axis}/{metric}/{statistic}'] = value
    output['coverage/failed_episodes'] = len(score['coverage']['failed_episodes'])
    output['coverage/missing_required_phases'] = len(score['coverage']['required_missing_phases'])
    return output


def selfcheck():
    """No model, dataset, optimizer or physics: exercise hierarchy and gate edges."""
    frames = np.arange(8*2*25,dtype=np.float32).reshape(8,2,25)
    history = rebuild_history(frames)
    assert np.array_equal(history[0],np.tile(frames[0],(1,5)))
    assert np.array_equal(history[-1],frames[-5:].transpose(1,0,2).reshape(2,125))
    assert list(OFFLINE_GATES['offline_vx_gates']['initial'].values()) == [.002,.005,.002,.005]
    # Exact comparisons at the declared boundary; small-bias exception applies to bias, never MAE.
    base = dict(run='sentinel',env=0,bucket='parking',segment=1,axis='x',start_s=0.,end_s=4.,
        expected_samples=400,status='complete',metrics=dict(mae_m_s=.02,p95_absolute_error_m_s=.05,
            bias_m_s=-.01,abs_bias_m_s=.01,peak_absolute_error_m_s=.1,peak_time_s=1.,band_1_3Hz_rms_m_s=.002))
    candidate = dict(base,metrics=dict(base['metrics'],mae_m_s=.016,abs_bias_m_s=.008))
    coverage = dict(failed_episodes=[],required_missing_phases=[],missing_complete_3s_parking_envs=[])
    a = dict(data_lock_sha256='sentinel',rows=[base],coverage=coverage)
    b = dict(data_lock_sha256='sentinel',rows=[candidate],coverage=coverage)
    assert candidate_gate(b,a)['passed']
    candidate['metrics']['mae_m_s'] = np.nextafter(.016,np.inf)
    assert not candidate_gate(b,a)['passed']
    missing = dict(base,status='missing',metrics=None)
    assert not candidate_gate(dict(b,rows=[missing]),a)['passed']
    return dict(verified=True,history_reset_padding=True,gate_boundary_exact=True,
        missing_phase_fails=True,training_updates=0,model_inference=False,physics=False)


if __name__ == '__main__':
    print(json.dumps(selfcheck(),ensure_ascii=False))
