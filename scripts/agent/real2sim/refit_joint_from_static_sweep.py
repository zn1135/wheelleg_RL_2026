"""Refit original dynamic recordings with the new static load envelope fixed; GPU only."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def canonical(node):
    return node.tag, sorted(node.attrib.items()), (node.text or '').strip(), [canonical(n) for n in node]


def prepare_reference(name, output):
    old, rc = map(Path, fit.REFERENCES[name])
    report = json.loads((old / 'report.json').read_text())
    current = Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml').resolve()
    frozen = Path('artifacts/model-versions') / report['source_model_sha256'] / 'source/model.xml'
    if not frozen.is_file():
        frozen = Path('artifacts/shared_validation/closed-chain-training-integration-20261104-01/before-source.xml')
    assert fit.audit.digest(frozen) == report['source_model_sha256']
    before, after = ET.parse(frozen).getroot(), ET.parse(current).getroot()
    reverted = copy.deepcopy(after)
    # 新共享XML只比旧录制版本收紧闭链/限位响应；其余几何、惯量和量程必须相同。
    for key in ('solreflimit', 'solimplimit'):
        reverted.find('default/joint').attrib.pop(key)
    for a, b in zip(before.find('equality'), reverted.find('equality')):
        for key in ('solref', 'solimp'):
            b.set(key, a.get(key))
    assert canonical(before) == canonical(reverted), 'Unexpected model changes beyond already-adopted constraint response'
    output.mkdir(parents=True, exist_ok=False)
    for name_ in ('replay.npz', 'fixed-base-model.xml'):
        shutil.copy2(old / name_, output / name_)
    shutil.copy2(rc, output / 'rc-inputs.npz')
    report.update(original_reference=str(old.resolve()), original_report_sha256=fit.audit.digest(old / 'report.json'),
                  recorded_source_model_sha256=report['source_model_sha256'],
                  source_model=str(current), source_model_sha256=fit.audit.digest(current),
                  diagnostic_model_migration='Only adopted loop/limit response settings; measured data and initial state unchanged')
    fit.search.write_json(output / 'report.json', report)
    fit.REFERENCES[name] = (str(output), str(output / 'rc-inputs.npz'))
    return fit.input_data(name)


def static_envelope(path):
    report = json.loads(path.read_text())
    ref = report['reference_compression_mm']
    ext_ref, ext_k, con_ref, con_k = report['results']['direction_linear']['full_fit']['parameters']
    center_ref = (ext_ref + con_ref) / 2 + (ext_k + con_k) / 2 * (5. - ref)
    center_k = (ext_k + con_k) * 500.
    friction_ref = (con_ref - ext_ref) / 2 + (con_k - ext_k) / 2 * (5. - ref)
    friction_k = (con_k - ext_k) * 500.
    grid = np.linspace(0., .06, 100)
    pressure = center_ref + center_k * (grid - .005)
    capacity = friction_ref + friction_k * (grid - .005)
    assert pressure.min() > 0 and capacity.min() >= 0
    assert np.allclose(pressure-capacity, ext_ref + ext_k*(grid*1000-ref), atol=1e-10)
    assert np.allclose(pressure+capacity, con_ref + con_k*(grid*1000-ref), atol=1e-10)
    return dict(spring_reference_force_n=[center_ref]*2, spring_stiffness_n_m=[center_k]*2,
                spring_friction_n=[friction_ref]*2, spring_friction_stiffness_n_m=[friction_k]*2,
                spring_force_limit_n=[600.]*2, armature_kg_m2=[.015795]*4)


def encode(profile, side):
    vector = []
    for i in (2*side, 2*side+1):
        static = profile['friction_nm'][i]
        vector.extend([fit.search.logarithm(static, .05, 5.),
                       (profile['motor_dynamic_positive_nm'][i]/static-.05)/.95,
                       (profile['motor_dynamic_negative_nm'][i]/static-.05)/.95,
                       fit.search.logarithm(profile['motor_velocity_scale_rad_s'][i], .002, .2),
                       fit.search.logarithm(profile['damping_nms_rad'][i], .01, 4.)])
    vector.append(fit.search.logarithm(profile['spring_damping_ns_m'][side], 2., 500.))
    return np.clip(vector, 0., 1.)


def decode(vectors, envelope):
    result = copy.deepcopy(envelope)
    for key in ('friction_nm', 'motor_dynamic_positive_nm', 'motor_dynamic_negative_nm',
                'motor_velocity_scale_rad_s', 'damping_nms_rad', 'spring_damping_ns_m'):
        result[key] = []
    for vector in vectors:
        for offset in (0, 5):
            static = .05*100.**vector[offset]
            result['friction_nm'].append(float(static))
            result['motor_dynamic_positive_nm'].append(float(static*(.05+.95*vector[offset+1])))
            result['motor_dynamic_negative_nm'].append(float(static*(.05+.95*vector[offset+2])))
            result['motor_velocity_scale_rad_s'].append(float(.002*100.**vector[offset+3]))
            result['damping_nms_rad'].append(float(.01*400.**vector[offset+4]))
        result['spring_damping_ns_m'].append(float(2.*250.**vector[10]))
    return result


def quality(reports):
    result = np.zeros(2)
    for report in reports:
        row = report['results'][0] if len(report['results']) == 1 else None
        assert row is not None
        for side in (0, 1):
            section = slice(2*side, 2*side+2)
            branches = [row[k] for k in ('recorded_torque', 'recorded_target')]
            if row['feedback_blocked'] or any(not b['original_assembly_branch'] or b['limit_excess_deg'] > .1 for b in branches):
                result[side] += 1e6
            else:
                rmse = np.array([b['rmse_deg'][section] for b in branches])
                peak = max(max(b['peak_deg'][section]) for b in branches)
                terminal = max(max(b['terminal_deg'][section]) for b in branches)
                result[side] += float(np.sqrt(np.mean(rmse**2)) + .05*peak + .1*terminal)
    return result / len(reports)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--static-fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--population', type=int, default=16)
    parser.add_argument('--generations', type=int, default=2)
    args = parser.parse_args()
    assert 8 <= args.population <= 32 and 0 <= args.generations <= 4
    args.output.mkdir(parents=True, exist_ok=False)
    protected_paths = [Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'),
                       Path('deployment/H7_RL_ros/imcalib/config/machine_config.c'),
                       Path(fit.prior.replay.training_spring.__file__)]
    protected = {str(p):fit.audit.digest(p) for p in protected_paths}
    envelope = static_envelope(args.static_fit)
    fit.search.write_json(args.output/'static-envelope.json', dict(parameters=envelope,
        source=str(args.static_fit.resolve()), source_sha256=fit.audit.digest(args.static_fit),
        formulas=['pressure=(extend+contract)/2', 'native_slide_friction_capacity=(contract-extend)/2'],
        conditional_not_physical_identification=True))
    inputs = {name:prepare_reference(name, args.output/'inputs'/name) for name in tuple(fit.REFERENCES)}
    fit.wp.init()
    baseline_path = Path('artifacts/real2sim/h7-ordered-refit-20261104-02-best/selected-parameters.json')
    baseline = json.loads(baseline_path.read_text())
    rng = np.random.default_rng(20261105)
    vectors = rng.uniform(size=(args.population, 2, 11))
    for i, static in enumerate((.4, .8, 1.5, 3.)):
        seed = copy.deepcopy(baseline)
        seed['friction_nm'] = [static]*4
        seed['motor_dynamic_positive_nm'] = [static*.5]*4
        seed['motor_dynamic_negative_nm'] = [static*.5]*4
        seed['damping_nms_rad'] = [.2]*4
        vectors[i] = [encode(seed, side) for side in (0, 1)]
    for i in range(4, args.population//2):
        vectors[i] = np.clip(vectors[i % 4]+rng.normal(0., .15, (2, 11)), 0., 1.)
    total, history = 0, []

    def evaluate(values, label):
        nonlocal total
        profiles = [decode(v, envelope) for v in values]
        fit.search.write_json(args.output/(label+'-parameters.json'), profiles)
        trials = [[] for _ in profiles]
        for name in ('original', 'wide'):
            _, report = fit.run(profiles, inputs[name], args.output/label/name, solver_profile='ordered')
            total += report['new_gpu_environment_steps']
            for i, row in enumerate(report['results']):
                trials[i].append(dict(results=[row]))
        return np.array([quality(rows) for rows in trials])

    scores = evaluate(vectors, 'generation-00')
    for generation in range(args.generations+1):
        if generation:
            proposals = vectors.copy()
            for i in range(args.population):
                for side in (0, 1):
                    a, b, c = rng.choice(np.delete(np.arange(args.population), i), 3, replace=False)
                    mutant = np.clip(vectors[a, side]+.65*(vectors[b, side]-vectors[c, side]), 0., 1.)
                    mask = rng.uniform(size=11)<.7; mask[rng.integers(11)] = True
                    proposals[i, side] = np.where(mask, mutant, vectors[i, side])
            trial_scores = evaluate(proposals, f'generation-{generation:02d}')
            # 固定机身、无接触使左右侧可独立选优；组合后仍单候选完整复算。
            improve = trial_scores < scores
            vectors[improve] = proposals[improve]; scores[improve] = trial_scores[improve]
        entry = dict(generation=generation, best_side_scores=scores.min(0).tolist(), gpu_environment_steps=total)
        history.append(entry); fit.search.write_json(args.output/'progress.json', history)
        print(json.dumps(entry), flush=True)
    selected = decode(np.array([vectors[np.argmin(scores[:, side]), side] for side in (0, 1)]), envelope)
    fit.search.write_json(args.output/'selected-parameters.json', selected)
    final, before = {}, {}
    for name in inputs:
        _, report = fit.run([selected], inputs[name], args.output/('final-'+name), full=True, solver_profile='ordered')
        final[name] = report['results'][0]; total += report['new_gpu_environment_steps']
        _, report = fit.run([baseline], inputs[name], args.output/('baseline-'+name), full=True, solver_profile='ordered')
        before[name] = report['results'][0]; total += report['new_gpu_environment_steps']
        fit.search.write_json(args.output/'partial-results.json', dict(final=final, baseline=before, gpu_environment_steps=total))
    assert {str(p):fit.audit.digest(p) for p in protected_paths} == protected
    fit.search.write_json(args.output/'report.json', dict(final=final, baseline=before, selected_parameters=selected,
        static_envelope=envelope, history=history, new_gpu_environment_steps=total, protected_sources=protected,
        baseline_parameters_sha256=fit.audit.digest(baseline_path), development_sessions=['original','wide'],
        regression_sessions=['slow','fast'], independent_hardware_holdout=False,
        shared_model_modified=False, hardware_access=False, policy_training_updates=0))
    for path in (Path(__file__), Path(fit.__file__), Path(fit.prior.__file__)):
        shutil.copy2(path, args.output/path.name)


if __name__ == '__main__':
    main()
