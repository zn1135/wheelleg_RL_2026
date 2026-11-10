"""Re-evaluate a frozen candidate pool with canonical inertia; reuse only audited identical wide runs."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); fit.wp.init()
    inputs = {name:fit.input_data(name) for name in fit.REFERENCES}
    canonical = mujoco.MjModel.from_xml_string(ET.tostring(fit.canonical_xml(inputs['wide']), encoding='unicode'))
    fields = ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat',
              'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited', 'eq_data', 'qpos0', 'site_pos', 'dof_armature']
    all_scores = []; all_profiles = []; provenance = []; new_steps = 0; reused_steps = 0
    for generation in range(4):
        label = f'generation-{generation:02d}'; previous = args.previous/label/'wide'
        wide = json.loads((previous/'report.json').read_text()); oldmodel = mujoco.MjModel.from_xml_path(str(previous/'model.xml'))
        assert all(np.array_equal(getattr(canonical, f), getattr(oldmodel, f)) for f in fields)
        assert fit.audit.digest(previous/'model.xml') == wide['model_sha256']
        for path, value in wide['input_sha256'].items(): assert fit.audit.digest(Path(path)) == value
        # 旧模板运行时已由归档脚本设为这组值；线搜索参数来自旧参考模板。
        oldmodel.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        oldmodel.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        oldmodel.opt.iterations = 100; oldmodel.opt.tolerance = 1e-10; oldmodel.opt.impratio = 10.
        assert oldmodel.opt.ls_iterations == 100 and oldmodel.opt.ls_tolerance == .0001
        assert oldmodel.opt.timestep == .0002 and oldmodel.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
        profiles = json.loads((args.previous/(label+'-parameters.json')).read_text())
        assert profiles == [r['parameters'] for r in wide['results']]
        scores, original = fit.run(profiles, inputs['original'], args.output/label/'original')
        joint = (scores+np.asarray(wide['side_scores']))/2
        all_scores.extend(joint); all_profiles.extend(profiles)
        new_steps += original['new_gpu_environment_steps']; reused_steps += wide['new_gpu_environment_steps']
        provenance.append(dict(generation=generation, reused_wide=str(previous),
            reused_report_sha256=fit.audit.digest(previous/'report.json'),
            reused_trace_sha256=fit.audit.digest(previous/'trials.npz'),
            physical_arrays_bitwise_equal_to_shared=fields, side_scores=joint.tolist()))
        fit.search.write_json(args.output/'progress.json', provenance)
        print(json.dumps(dict(corrected_generation=generation, best_side_scores=np.asarray(all_scores).min(0).tolist())), flush=True)
    scores = np.asarray(all_scores); ix = np.argmin(scores, axis=0)
    selected = {}
    for key in all_profiles[0]:
        left, right = all_profiles[ix[0]][key], all_profiles[ix[1]][key]
        if len(left) == 2: selected[key] = left[:1]+right[1:]
        elif len(left) == 4: selected[key] = left[:2]+right[2:]
        else: raise ValueError(key)
    fit.search.write_json(args.output/'selected-parameters.json', selected); final = {}
    for name in fit.REFERENCES:
        _, result = fit.run([selected], inputs[name], args.output/('final-'+name), full=True)
        final[name] = result['results'][0]; new_steps += result['new_gpu_environment_steps']
    protected = fit.audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'), [])
    fit.search.write_json(args.output/'report.json', dict(history=provenance, selected_parameters=selected, final=final,
        new_gpu_environment_steps=new_steps, reused_valid_gpu_environment_steps=reused_steps,
        verified_gpu_environment_steps=new_steps+reused_steps, protected_source_checks=protected,
        accepted=False, shared_model_modified=False, policy_training_updates=0, hardware_access=False,
        canonical_shared_inertia_for_all_sessions=True, frozen_candidates_re_evaluated=len(all_profiles),
        invalid_previous_original_scoring=str(args.previous/'invalidated.json'),
        selection='Equal-trajectory joint RMSE + peak and terminal penalties; independent-side candidates combined then re-run on all4 sessions'))
    for path in (Path(__file__), Path(fit.__file__)):
        shutil.copy2(path, args.output/path.name)


if __name__ == '__main__': main()
