#!/usr/bin/env python3
"""Offline continuous drive/coast candidate; never opens a hardware connection."""
from pathlib import Path
import argparse
import ctypes
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

import calibrate_fixed_h7_huananhu as previous
import real2sim_dynamics_huananhu as dynamics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'artifacts/fixed-h7-fixed300-data-20261021'
OLD = DATA/'fit-budget80'
GROUPS = ('orange_losses', 'coast_losses')
BOUNDS = np.array([[.25, 30., .5], [6., 180., 4.]])


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def load_data_module():
    spec = importlib.util.spec_from_file_location('complete_capture_fit', DATA/'fit_capture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_gain(out):
    """Use MuJoCo's native user-gain callback inside its actual integrator."""
    package = Path(mujoco.__file__).parent
    compiler = os.environ.get('CC', '/home/zn/miniforge3/envs/fudan_env/bin/x86_64-conda-linux-gnu-gcc')
    command = [compiler, '-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Werror',
               '-I'+str(package/'include'), str(ROOT/'sim2sim/h7_drive_response.c'),
               '-lm', '-o', str(out/'drive_gain.so')]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    save(out/'native-build.json', {'command':command, 'stdout':result.stdout, 'stderr':result.stderr})
    library = ctypes.CDLL(str(out/'drive_gain.so'))
    function = library.h7_drive_gain
    function.restype = ctypes.c_double
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    mujoco.set_mjcb_act_gain(function)
    return library


def configure(model, actuators, gain):
    for actuator, values in zip(actuators, gain):
        model.actuator_gaintype[actuator] = mujoco.mjtGain.mjGAIN_USER
        model.actuator_gainprm[actuator, :3] = values


def bench(pulse, losses, gain):
    configure(pulse.fixture.m, pulse.fixture.a, gain)
    return previous.fast(pulse, losses)


def coast(pulse, losses):
    """Fit-only loss constraint starts from measured off speed, never validation.

    Full-pulse predictions always use bench(), which never resets at off.
    """
    fixture = pulse.fixture
    fixture.parameters(losses)
    mujoco.mj_resetData(fixture.m, fixture.d)
    at = pulse.off
    fixture.d.qpos[fixture.q] = pulse.q0
    fixture.d.qvel[fixture.v] = 0
    fixture.d.qvel[fixture.v[pulse.side]] = np.interp(at, pulse.time, pulse.actual_v)
    duration = pulse.time[pulse.dec].max()-at
    steps = int(np.ceil(duration/pulse.dt))+1
    initial = np.empty(mujoco.mj_stateSize(fixture.m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(fixture.m, fixture.d, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states, _ = previous.rollout(fixture.m, fixture.d, initial, np.zeros((steps,fixture.m.nu)))
    states = np.vstack([initial, states[0]])
    velocity = states[:,1+fixture.m.nq+fixture.v[pulse.side]]
    return np.interp(pulse.time[pulse.dec]-at, np.arange(steps+1)*pulse.dt, velocity)


def bounded_fit(fun, starts, lower, upper, budget, scale, label):
    calls, best, cost, trace = 0, None, np.inf, []
    limit = budget
    class Done(Exception):
        pass
    def objective(x):
        nonlocal calls, best, cost
        if calls >= limit:
            raise Done()
        residual = fun(x)
        score = float(residual@residual)
        calls += 1
        trace.append({'call':calls, 'parameters':np.asarray(x).tolist(), 'cost':score})
        if score < cost:
            best, cost = np.array(x), score
        return residual
    for k, start in enumerate(starts):
        limit = (k+1)*(budget//len(starts))
        try:
            least_squares(objective, start, bounds=(lower,upper), x_scale=scale,
                          max_nfev=budget, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        except Done:
            pass
    limit = budget
    while calls < budget:
        objective(best)
    print(label, calls, best.tolist(), cost, flush=True)
    return best, {'calls':calls, 'cost':cost, 'trace':trace}


def fit_group(pulses, orange, group, budget):
    losses = orange.copy()
    details = {'losses':[], 'drive':[]}
    if group == 'coast_losses':
        for side in (0,1):
            ps = [p for p in pulses if p.side == side]
            def error(x):
                parameters = losses.copy()
                parameters[side,1:] = x
                return np.concatenate([(coast(p,parameters)-p.actual_v[p.dec])/np.sqrt(p.dec.sum()) for p in ps])
            x, log = bounded_fit(error, [np.array([.01,.05]),np.array([.003,.3])],
                                 np.zeros(2), np.array([.08,2.]), budget, np.array([.02,.4]),
                                 f'{group} coast {side}')
            losses[side,1:] = x
            details['losses'].append(log)
    gain = np.array([[1.,80.,1.],[1.,80.,1.]])
    for side in (0,1):
        ps = [p for p in pulses if p.side == side]
        def error(x):
            candidate = gain.copy()
            candidate[side] = x
            pieces = []
            for p in ps:
                _,v = bench(p,losses,candidate)
                residual = p.sampled(v)-p.actual_v
                pieces.extend([residual[p.acc]/np.sqrt(p.acc.sum()),residual[p.dec]/np.sqrt(p.dec.sum())])
            return np.concatenate(pieces)
        x, log = bounded_fit(error, [np.array([3.,75.,1.]),np.array([1.5,100.,2.])],
                             BOUNDS[0], BOUNDS[1], budget, np.array([2.,60.,1.]),
                             f'{group} drive {side}')
        gain[side] = x
        details['drive'].append(log)
    return {'additional_parameters':losses.tolist(), 'drive_gain':gain.tolist(), 'optimization':details}


def export(model_path, path, item):
    dynamics.export_candidate(model_path,path,np.asarray(item['additional_parameters']))
    # MjSpec.to_xml rounds unrelated geometry to six significant digits, which
    # breaks this model's tight closed-chain pose solve. Preserve original text.
    tree = ET.parse(path)
    for name, values in zip(dynamics.mirror.WHEELS,item['drive_gain']):
        actuator = tree.find(f".//actuator/motor[@name='{name}_motor']")
        actuator.tag = 'general'
        actuator.set('gaintype','user')
        actuator.set('gainprm',' '.join(format(x,'.17g') for x in values))
    tree.write(path,encoding='utf-8',xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(path))


def predict_rows(module,pulses,models,split):
    rows = []
    for p in pulses:
        for name,item in models.items():
            _,v = bench(p,np.asarray(item['additional_parameters']),np.asarray(item['drive_gain']))
            pred = p.sampled(v)
            for i,t in enumerate(p.time):
                phase = 'acceleration' if p.acc[i] else 'deceleration' if p.dec[i] else 'edge_or_initial'
                speed = abs(p.actual_v[i])
                band = next(label for lo,hi,label in [(0,3,'0-3'),(3,10,'3-10'),(10,30,'10-30'),(30,50,'30-50'),(50,80,'50-80'),(80,np.inf,'80+')] if lo <= speed < hi)
                rows.append({'split':split,'session':p.session,'model':name,'pulse':p.info['number'],
                             'side':p.info['side'],'direction':1 if p.info['command_nm']>0 else -1,
                             'phase':phase,'speed_band':band,'time_from_on_s':float(t-p.on),
                             'actual_rad_s':float(p.actual_v[i]),'predicted_rad_s':float(pred[i]),
                             'error_rad_s':float(pred[i]-p.actual_v[i])})
    return rows


def smoke(out,module,orange):
    _,pulses = module.load('fit-complete')
    p = pulses[0]
    gain = np.array([[2.,80.,1.],[2.,80.,1.]])
    q,v = bench(p,orange,gain)
    # Independent Python force-law stepping verifies the native callback.
    f = p.fixture
    f.m.actuator_gaintype[f.a] = mujoco.mjtGain.mjGAIN_FIXED
    f.m.actuator_gainprm[f.a,0] = 1.
    mujoco.mj_resetData(f.m,f.d)
    f.d.qpos[f.q],f.d.qvel[f.v] = p.q0,p.v0
    qs,vs = [p.q0.copy()],[p.v0.copy()]
    for u in p.u:
        speed = f.d.qvel[f.v]
        along = np.sign(u)*speed
        g = gain[:,0]*np.maximum(0,1-np.maximum(0,along)/gain[:,1])**gain[:,2]
        f.d.ctrl[f.a] = u*g
        # ctrlrange is nominal-request range; map gain through fixed actuators
        # only for this independent comparison, so temporarily widen the range.
        original_range = f.m.actuator_ctrlrange[f.a].copy()
        f.m.actuator_ctrlrange[f.a] = [-100.,100.]
        mujoco.mj_step(f.m,f.d)
        f.m.actuator_ctrlrange[f.a] = original_range
        qs.append(f.d.qpos[f.q].copy());vs.append(f.d.qvel[f.v].copy())
    difference = max(float(np.max(abs(q-np.asarray(qs)))),float(np.max(abs(v-np.asarray(vs)))))
    assert difference < 1e-8, difference
    configure(f.m,f.a,gain)
    a,b = bench(p,orange,gain)
    measured = p.actual_v.copy();p.actual_v[:] = 1e8
    aa,bb = bench(p,orange,gain);p.actual_v[:] = measured
    assert np.array_equal(a,aa) and np.array_equal(b,bb)
    at = int(round(p.off/p.dt))
    assert np.isfinite(v).all() and abs(v[at,p.side]-v[at-1,p.side]) < 5
    # A zero-command coast uses the same passive losses, regardless of drive gain.
    zero = p.u.copy();p.u[:] = 0
    p.v0[:] = [40.,-30.]
    _,z1 = bench(p,orange,gain)
    _,z2 = bench(p,orange,gain*[[1.5,1.,1.],[1.5,1.,1.]])
    assert np.array_equal(z1,z2)
    p.u[:] = zero;p.v0[:] = 0
    save(out/'smoke.json',{'passed':True,'native_gain_vs_independent_python_max_abs':difference,
                          'future_measured_speed_does_not_affect_prediction':True,
                          'zero_input_same_passive_coast':True,'off_speed_not_reset':True,
                          'scope':'one fit pulse, continuous MuJoCo integration; no physical acceptance'})


def rl_run(out,models,small=False):
    """Use identical old held-out starts/inputs; these remain coarse-input diagnostics."""
    s = dynamics.Session(ROOT/previous.RL)
    old = json.loads((OLD/'rl.json').read_text())
    saved = np.load(OLD/'rl-states.npz')
    data = dynamics.RecordedRL(s)
    # Existing input and contact sensitivities, unchanged from the comparison.
    cases = [('lower_contact','zoh','lower',-1e-6),('zero_translation','zoh','zero',-1e-6),
             ('floor_plus_1mm','zoh','lower',.001),('friction_055','zoh','lower',-1e-6),
             ('linear_torque','linear','lower',-1e-6),('pitch_plus_01deg','zoh','lower',-1e-6),
             ('quantized_wheel_torque','quantized','lower',-1e-6),('initial_wheel_plus_01','zoh','lower',-1e-6)]
    predictors = {n:dynamics.BodyPredictor(out/(n+'.xml'),s.config) for n in models}
    rows = []
    starts = old['initialization'][:2] if small else old['initialization']
    for k,initial in enumerate(starts):
        i = initial['row']
        for case,method,translation,gap in cases:
            for name,predictor in predictors.items():
                q,v = saved[f'q_{i}'].copy(),saved[f'v_{i}'].copy()
                q[2] += gap
                if translation == 'lower':v[:3] = initial['lower_velocity']
                if case == 'pitch_plus_01deg':mujoco.mju_quatIntegrate(q[3:7],np.array([0.,np.deg2rad(.1),0.]),1.)
                if case == 'initial_wheel_plus_01':v[predictor.env.v[dynamics.sim.WHEELS]] += .1
                pq,pv = previous.torque_predict(predictor,q,v,s,i,method,case)
                for h in (10,50,100):
                    j,step = i+h//10,h//2
                    error = pv[step,predictor.env.v][dynamics.N2I][4:]-s.f[j,130:132]
                    rows.append({'row':i,'case':case,'model':name,'horizon_ms':h,'left_error':float(error[0]),
                                 'right_error':float(error[1]),'pitch_error_deg':float(np.rad2deg(dynamics.pitch(pq[step,3:7])-dynamics.pitch(data.attitude(s.output_ms[i]+h))))})
        if k%30 == 0:print('RL',k+1,'/',len(starts),flush=True)
    previous.write_csv(out/('rl-small.csv' if small else 'rl-predictions.csv'),rows)
    if not small:
        save(out/'rl.json',{'windows':len(starts),'cases':len(cases),'endpoints':len(rows),
                           'metrics':{col:previous.summarize(rows,['case','model','horizon_ms'],col)
                                      for col in ('left_error','right_error','pitch_error_deg')},
                           'scope':'historical100Hz-request, unknown human support, translation unmeasured; not complete-input RL acceptance',
                           'gain_outside_3Nm_motoring_quadrant':'unvalidated linear command scaling and low-speed-gain braking extension'})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--budget',type=int,default=100)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--resume-evaluation',action='store_true',help='Reuse frozen fit parameters after an interrupted evaluation; no refitting')
    a = p.parse_args();a.out.mkdir(parents=True,exist_ok=a.resume_evaluation)
    module = load_data_module()
    original = json.loads((OLD/'report.json').read_text())
    orange = np.asarray(original['models']['free_armature']['additional_parameters'])
    library = install_gain(a.out)
    try:
        smoke(a.out,module,orange)
        if a.smoke_only:return
        s,train = module.load('fit-complete')
        if a.resume_evaluation:
            models = json.loads((a.out/'parameters.json').read_text())['models']
        else:
            models = {name:fit_group(train,orange,name,a.budget) for name in GROUPS}
        for name,item in models.items():export(s.model_path,a.out/(name+'.xml'),item)
        if not a.resume_evaluation:save(a.out/'parameters.json',{'models':models,'gain_order':['gain_at_zero_speed','speed_knee_rad_s','exponent'],
             'gain_bounds':BOUNDS.tolist(),'loss_bounds':[[0,0],[.08,2.]],'budget_each_stage_per_wheel':a.budget,
             'fit_session':'fit-complete','validation_session':'validation-complete','same_inertia_drive_and_coast':True,
             'validation_not_used_by_optimizer':True,'note':'previous holdout curves were already inspected; repeated reuse is not a new blind experiment'})
        # Only expand after the finite native/continuous-step smoke checks.
        _,valid = module.load('validation-complete')
        rows = predict_rows(module,train,models,'fit')+predict_rows(module,valid,models,'validation')
        previous.write_csv(a.out/'predictions.csv',rows)
        _,fine = module.load('validation-complete',dt=.0005)
        fine_rows = predict_rows(module,fine,models,'validation_dt0p5ms')
        previous.write_csv(a.out/'fine-predictions.csv',fine_rows)
        interior = [r for r in rows if r['phase'] != 'edge_or_initial']
        save(a.out/'bench.json',{'aggregate':module.summarize(interior,['split','model']),
             'phase_summary':module.summarize(interior,['split','model','side','direction','phase']),
             'speed_summary':module.summarize(interior,['split','model','side','direction','phase','speed_band']),
             'half_step_aggregate':module.summarize([r for r in fine_rows if r['phase'] != 'edge_or_initial'],['split','model']),
             'whole_curve_aggregate':module.summarize(rows,['split','model']),
             'automatic_adoption':False})
        save(a.out/'run.json',{'complete':True,'command':sys.argv,'mujoco':mujoco.__version__,
             'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [Path(__file__),ROOT/'sim2sim/h7_drive_response.c']}})
    finally:
        mujoco.set_mjcb_act_gain(None)


if __name__ == '__main__':
    main()
