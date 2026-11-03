"""Read-only original parallel-robot/ONNX zero hold, bounded GPU replay and video."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET

import numpy as np

REFERENCE = Path('/home/zn/文档/fudan_rl_wheel_leg-main')
RUNTIME = REFERENCE / 'mujoco/python_tools/onnx_mj_binglian.py'
XML = REFERENCE / 'mujoco/assert_now/infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml'
ONNX = REFERENCE / 'mujoco/actor/yuntai/上台阶3_angz+.onnx'
DURATION = 15.


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference():
    # 仅移除桌面键盘依赖；观测、历史、策略和控制函数直接使用原文件。
    tree = ast.parse(RUNTIME.read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == 'pynput')]
    module = types.ModuleType('fudan_original_parallel_runtime')
    module.__file__ = str(RUNTIME)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(RUNTIME), 'exec'), module.__dict__)
    return module


def check(output):
    from unittest.mock import patch
    import torch
    output.mkdir(parents=True, exist_ok=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('No model loading, CUDA or physics in preparation')
    names = []
    def verify(name, condition):
        assert condition, name
        names.append(name)
    with patch('torch.cuda._lazy_init', side_effect=forbidden), patch('mujoco.MjModel.from_xml_path', side_effect=forbidden), \
            patch('onnxruntime.InferenceSession', side_effect=forbidden), patch('mujoco.mj_step', side_effect=forbidden):
        ref = load_reference()
        verify('default_assets_match_reference', Path(ref.DEFAULT_XML_PATH) == XML and Path(ref.DEFAULT_ONNX_PATH) == ONNX)
        original = ast.parse(RUNTIME.read_text())
        expected = next(n for n in original.body if isinstance(n, ast.ClassDef) and n.name == 'BinglianRuntime')
        import inspect
        actual = ast.parse(inspect.getsource(ref.BinglianRuntime)).body[0]
        verify('original_runtime_class_unchanged', ast.dump(expected) == ast.dump(actual))
        runtime = object.__new__(ref.BinglianRuntime)
        runtime.args = types.SimpleNamespace(print_base_ang_vel=False)
        runtime.command_state = ref.CommandState()
        runtime.active_command_scale = ref.PLANE_COMMAND_SCALE.copy()
        runtime.active_default_obs_dof_pos = ref.PLANE_DEFAULT_DOF_POS[ref.OBS_DOF_POS_IDXS]
        runtime.last_actions = np.zeros(6, np.float32)
        runtime.history = None
        obs = runtime.build_obs(np.zeros(3), np.array([0., 0., -1.]), ref.PLANE_DEFAULT_DOF_POS, np.zeros(6))
        expected_command = np.array([0., 0., .12], np.float32) * np.array([3., .25, 5.], np.float32)
        verify('obs25_zero_speed_effective_height012', obs.shape == (25,) and np.array_equal(obs[6:9], expected_command))
        runtime.update_history(obs)
        verify('first_history_repeats_same_frame5', runtime.history.shape == (5, 25) and np.array_equal(runtime.history, np.tile(obs, (5, 1))))
        old = runtime.history.copy()
        second = obs.copy(); second[0] = .2
        runtime.update_history(second)
        verify('later_history_advances_once', np.array_equal(runtime.history[:4], old[1:]) and np.array_equal(runtime.history[-1], second))
        class SessionProbe:
            def run(self, outputs, feed):
                assert feed['obs'].shape == (1, 25) and feed['hist'].shape == (1, 125)
                return [np.array([[101., -101., 1., 2., 3., 4.]], np.float32)]
        runtime.sess = SessionProbe(); runtime.obs_name = 'obs'; runtime.hist_name = 'hist'
        action = runtime.run_policy(second)
        verify('original_action_clip_and_previous_action', np.array_equal(action, [100., -100., 1., 2., 3., 4.]) and np.array_equal(runtime.last_actions, action))
        verify('original_controller_defaults', np.array_equal(ref.PLANE_P_GAINS, [15., 15., 0., 15., 15., 0.]) and ref.LEFT_GAS_SPRING_CTRL == ref.RIGHT_GAS_SPRING_CTRL == 300.)
        verify('reference_source_does_not_assign_timestep', not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'timestep' for t in n.targets) for n in ast.walk(original)))
    files = {RUNTIME, XML, ONNX, Path(__file__).resolve()}
    for node in ET.parse(XML).getroot().findall('./asset/*'):
        if node.get('file'):
            files.add((XML.parent / node.get('file')).resolve())
    verify('all_referenced_assets_exist', all(p.is_file() for p in files))
    manifest = {str(p): digest(p) for p in sorted(files)}
    for path in files:
        relative = path.relative_to(REFERENCE) if path.is_relative_to(REFERENCE) else Path('probe_reference_zero.py')
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(output / 'source-sha256.json', manifest)
    write(output / 'checks.json', dict(passed=True, checks=names, physics_steps=0, model_forwards=0, cuda_initializations=0))
    print(json.dumps(dict(passed=True, checks=len(names), source_files=len(files))), flush=True)


def guard(output, stage):
    budget = json.loads((output / 'budget.json').read_text())
    limit = budget['replay_stop_unix'] - 10 if stage == 'replay' else budget['deadline_unix'] - 5
    if time.time() >= limit:
        raise TimeoutError(stage + ' deadline')


def replay(output):
    import mujoco
    import mujoco_warp as mw
    import warp as wp
    ref = load_reference()
    args = ref.build_arg_parser().parse_args([])
    args.device = 'cpu'  # 原ONNX直接交给ORT，不转换权重；物理仍为CUDA。
    state = ref.CommandState()
    runtime = ref.BinglianRuntime(args, state)
    model, data = runtime.m, runtime.d
    dt = float(model.opt.timestep)
    assert abs(dt - .002) < 1e-12, '2ms trace adapter requires actual XML timestep=0.002; do not silently change it'
    count = round(DURATION / dt)
    policy_every = runtime.steps_per_policy
    effective_height = float(np.clip(state.cmd_height, ref.HEIGHT_MIN, ref.HEIGHT_MAX))
    case = dict(id='original_parallel_zero', category='zero', duration=DURATION,
                segments=[dict(start=0., duration=DURATION, command=[0., 0., effective_height])])
    sample = dict(id='reference_zero', case=case, randomized=False, seed=0)
    write(output / 'protocol.json', dict(protocol='fudan-original-parallel-onnx-zero-v1', mode='reference_zero_hold',
        robot='original_infantry_V2_parallel', validation_backend='mujoco_warp_gpu', samples=[sample],
        diagnostic_windows=[[2., 7.], [10., 15.]], unified_measurement_window=[2., 15.]))
    (output / 'fixtures').mkdir(); (output / 'traces').mkdir()
    mujoco.mj_saveModel(model, str(output / 'fixtures/model.mjb'))
    np.savez_compressed(output / 'fixtures/reference_zero_initial.npz', qpos=data.qpos.copy(), qvel=data.qvel.copy(),
        body_mass=model.body_mass.copy(), body_ipos=model.body_ipos.copy())
    metadata = dict(robot='original_infantry_V2_parallel', onnx=str(ONNX), xml=str(XML), runtime=str(RUNTIME),
        inference_providers=runtime.sess.get_providers(), inputs=[dict(name=i.name, shape=i.shape) for i in runtime.sess.get_inputs()],
        declared_sim_dt=args.sim_dt, declared_policy_dt=args.policy_dt, actual_xml_dt=dt,
        physics_steps_per_policy=policy_every, actual_policy_dt=policy_every * dt,
        original_height=state.cmd_height, effective_height=effective_height,
        p_gains=runtime.active_p_gains.tolist(), d_gains=runtime.active_d_gains.tolist(),
        nominal=runtime.active_default_dof_pos.tolist(), command_scale=runtime.active_command_scale.tolist(),
        mujoco_version=mujoco.__version__, physics_backend='mujoco_warp_gpu',
        source_logic='Original runtime methods, only desktop keyboard import removed; no original files edited',
        state_refresh='Original mj_step phase preserved: sensors/xquat are not refreshed by an extra forward after integration')
    write(output / 'runtime.json', metadata)
    wp.init()
    rows = []; poses = []; records = []; error = None; steps = 0; calls = 0
    base = runtime.handles.base_bid
    wheels = [model.body(n).id for n in ('l_wheel_Link', 'r_wheel_Link')]
    weight = float(model.body_subtreemass[base] * abs(model.opt.gravity[2]))
    with wp.ScopedDevice('cuda:0'):
        gm = mw.put_model(model)
        gd = mw.put_data(model, data, nworld=1, nconmax=512, njmax=2048)
        ids = wp.array(np.arange(512, dtype=np.int32), dtype=wp.int32)
        forces = wp.zeros(512, dtype=wp.spatial_vector)
        guard(output, 'replay')
        with wp.ScopedCapture() as capture:
            mw.step(gm, gd)
            mw.contact_force(gm, gd, ids, True, forces)
        graph = capture.graph
        verify_time = float(gd.time.numpy()[0])
        assert verify_time == 0., 'Graph capture must not advance experimental state'
        def sync():
            for field in ('qpos', 'qvel', 'xpos', 'xquat', 'sensordata'):
                getattr(data, field)[:] = getattr(gd, field).numpy()[0]
            data.time = float(gd.time.numpy()[0])
        def record():
            q = data.qpos.copy(); velocity = data.qvel[:3].copy()
            rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, q[3:7]); rotation = rotation.reshape(3, 3)
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            pitch = np.arcsin(np.clip(-rotation[2, 0], -1., 1.))
            contact_count = int(gd.nacon.numpy()[0])
            if contact_count > 512 or int(gd.nefc.numpy()[0]) > 2048:
                raise RuntimeError('Contact/constraint capacity exceeded')
            support = np.zeros(2); base_force = np.zeros(3)
            if steps:
                pairs = gd.contact.geom.numpy()[:contact_count]
                contact_forces = forces.numpy()[:contact_count, :3]
                for pair, force in zip(pairs, contact_forces):
                    for geom, sign in ((int(pair[0]), -1.), (int(pair[1]), 1.)):
                        if geom < 0: continue
                        body = int(model.geom_bodyid[geom])
                        if body in wheels: support[wheels.index(body)] += sign * force[2]
                        if body == base: base_force += sign * force
            row = np.r_[q[:3], rotation.T @ velocity, velocity, data.qvel[3:6],
                        np.rad2deg([roll, pitch]), np.maximum(support, 0.), np.linalg.norm(base_force), weight]
            if not np.isfinite(row).all() or not np.isfinite(q).all(): raise FloatingPointError('Non-finite reference state')
            rows.append(row); poses.append(q)
        record()
        try:
            for tick in range(count // policy_every):
                guard(output, 'replay')
                state.update_ramp(runtime.eff_policy_dt, args.cmd_ramp_time)
                ang, gravity, q, qd = runtime.build_state()
                obs = runtime.build_obs(ang, gravity, q, qd)
                runtime.update_history(obs)
                actions = runtime.run_policy(obs); calls += 1
                if not np.isfinite(actions).all(): raise FloatingPointError('Non-finite original ONNX output')
                records.append(np.r_[steps * dt, obs, runtime.history.reshape(-1), actions])
                for substep in range(policy_every):
                    control = runtime.compute_control(actions, q, qd)
                    runtime.apply_ctrl(control)
                    if not np.isfinite(data.ctrl).all(): raise FloatingPointError('Non-finite original mapped control')
                    gd.ctrl.assign(np.asarray(data.ctrl[None], np.float32))
                    wp.capture_launch(graph); sync(); steps += 1; record()
                    if substep < policy_every - 1: _, _, q, qd = runtime.build_state()
                if tick % 500 == 0: print(json.dumps(dict(step=steps, total=count, simulation_s=steps * dt)), flush=True)
        except Exception as exc:
            error = repr(exc)
        finally:
            from ct_mjlab.fudan_ppo_walk.measurement import arrays
            trace = arrays(np.asarray(rows), np.asarray(poses)[::5])
            trace['qpos'] = np.asarray(poses)
            trace['qpos_time'] = np.arange(len(poses)) * dt
            trace['policy_records'] = np.asarray(records)
            trace['terminated'][:] = False
            if error: trace['numerical_failure'][-1] = True
            np.savez_compressed(output / 'traces/reference_zero.npz', **trace)
            execution = dict(engine='mjlab', validation_backend='mujoco_warp_gpu', complete=steps==count and error is None,
                physics_steps=steps, inference_calls=calls, optimizer_updates=0, error=error,
                last_physics_unix=time.time(), no_retry=True, backend_graph_capture_steps=0)
            write(output / 'execution.json', execution)
    if error: raise RuntimeError(error)


def finish(output):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from scripts.agent.fudan_ppo_walk.report_parking_integral import parking_metrics
    guard(output, 'finish')
    trace_path = output / 'traces/reference_zero.npz'
    if not trace_path.exists():
        write(output / 'result.json', dict(complete=False, reason='No trajectory; inspect replay.log', final_acceptance='not_ready'))
        return
    with np.load(trace_path) as saved: trace = {k:saved[k] for k in saved.files}
    metrics = [parking_metrics(trace, lo, hi) for lo, hi in ((2.,7.),(10.,15.))]
    write(output / 'zero-metrics.json', metrics)
    from scripts.agent.motion.acceptance import canonical
    report = canonical(output, 'fudan_ppo_walk')
    report['reference_robot'] = 'Original parallel robot and ONNX; not the local serial physical robot'
    (output / 'acceptance').mkdir()
    write(output / 'acceptance/acceptance.json', report)
    model = mujoco.MjModel.from_binary_path(str(output / 'fixtures/model.mjb'))
    data = mujoco.MjData(model)
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth=960; model.vis.global_.offheight=640
    camera=mujoco.MjvCamera(); camera.distance=1.6; camera.elevation=-18; camera.azimuth=100
    video=output/'zero-hold.mp4'; frames=0
    with mujoco.Renderer(model,height=640,width=960) as renderer, imageio.get_writer(video,fps=50,codec='libx264',quality=7) as writer:
        for index in range(0,len(trace['qpos'])-1,10):
            guard(output,'finish')
            data.qpos[:]=trace['qpos'][index]; mujoco.mj_kinematics(model,data)
            camera.lookat[:]=data.qpos[:3]; camera.lookat[2]+=.05
            renderer.update_scene(data,camera=camera)
            panel=Image.fromarray(renderer.render().copy()); draw=ImageDraw.Draw(panel)
            draw.rectangle((0,0,960,62),fill='#18212b')
            draw.text((8,7),f'Original parallel robot + original plane ONNX | Warp GPU | 1x | t={index*.002:.2f}s',fill='white')
            draw.text((8,28),f'vx_cmd=0 yaw_cmd=0 height_cmd=0.12 | vx={trace["linear_body"][index,0]:+.4f}m/s | x={trace["position"][index,0]:+.3f}m',fill='white')
            writer.append_data(np.asarray(panel)); frames+=1
    reader=imageio.get_reader(video); decoded=reader.count_frames(); reader.get_data(decoded-1); fps=reader.get_meta_data()['fps']; reader.close()
    result=dict(complete=len(trace['time'])==7501,metrics=metrics,unified_technical_passed=report['technical_passed'],
        video=dict(path=str(video),frames=frames,decoded_frames=decoded,fps=fps,complete=frames==decoded==750 and fps==50,sha256=digest(video)),
        physics_backend='mujoco_warp_gpu',final_acceptance='pending_user_video_review',last_compute_unix=time.time())
    write(output/'result.json',result)
    print(json.dumps(result),flush=True)


def run(output):
    manifest=json.loads((output/'source-sha256.json').read_text())
    assert all(digest(Path(p))==h for p,h in manifest.items())
    assert json.loads((output/'checks.json').read_text())['passed']
    assert not (output/'budget.json').exists(), 'A started budget cannot be reused'
    start=time.time(); mono=time.monotonic()
    write(output/'budget.json',dict(start_unix=start,replay_stop_unix=start+360,deadline_unix=start+600,
        maximum_seconds=600,authorization='用户选择原并联闭链+默认ONNX，最多600秒；无训练、无重试追加'))
    env=os.environ.copy(); env['MUJOCO_GL']='egl'; env['PYTHONPATH']='mjlab_training/src'
    stages=[]
    for name,limit in (('replay',360),('finish',600)):
        remaining=limit-(time.monotonic()-mono)
        if remaining<=0: stages.append(dict(stage=name,skipped='deadline')); continue
        with (output/(name+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','scripts.agent.fudan_ppo_walk.probe_reference_zero',name,'--output',str(output)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try: code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out=True; os.killpg(child.pid,signal.SIGKILL); code=child.wait()
        stages.append(dict(stage=name,exit_code=code,timed_out=timed_out,elapsed_s=time.monotonic()-mono))
        write(output/'progress.json',stages); print(json.dumps(stages[-1]),flush=True)
    write(output/'completion.json',dict(stages=stages,wall_s=time.monotonic()-mono,no_retry=True,
        original_files_unchanged=all(digest(Path(p))==h for p,h in manifest.items()),optimizer_updates=0))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=['check','run','replay','finish'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.mode](args.output)


if __name__=='__main__':
    main()
