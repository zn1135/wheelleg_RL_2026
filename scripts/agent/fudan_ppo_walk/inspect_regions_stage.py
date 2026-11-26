"""Frozen regional-training stage: original flat regression and paired terrain routes."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts.agent.fudan_ppo_walk import inspect_randomization_4000 as frozen

ROOT = frozen.ROOT
RUN = ROOT/'logs/mjlab/fudan_ppo_walk/20261116_regions10to20_resume5000_to8000_01'
EXPERIMENT = ROOT/'artifacts/fudan_ppo_walk/20261116-regions-resume5000-8000-01'
BASE = ROOT/'artifacts/fudan_ppo_walk/20261116-high-yaw5000-effect-01'
SOURCE = ROOT/'logs/mjlab/fudan_ppo_walk/20261116_high_yaw10_resume4000_to5000_01/model_final.pt'


def audit(out, iteration):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    source = RUN/f'model_{iteration}.pt'
    saved = torch.load(source, map_location='cpu', weights_only=False)
    assert saved['iterations'] == iteration and not saved['partial_update']
    assert saved['transitions'] == iteration*4096*48+184320
    assert saved['primary_updates'] == saved['encoder_updates'] == iteration*20
    for key in ('optimizer_state_dict', 'encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in saved[key]['state'].values()} == {iteration*20}
        assert all(torch.isfinite(v).all() for state in saved[key]['state'].values()
                   for v in state.values() if isinstance(v, torch.Tensor))
    assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
    b = json.loads((EXPERIMENT/'budget.json').read_text())
    assert saved['terrain'] == b['terrain'] and saved['terrain']['slope_degrees'] == [10., 20.]
    assert saved['resume_from']['sha256'] == b['checkpoint_sha256']
    assert json.loads((EXPERIMENT/'initialization-audit.json').read_text())['passed']
    events = EventAccumulator(str(RUN), size_guidance={'scalars': 0}).Reload()
    assert [v.step for v in events.Scalars('Train/reward') if v.step <= iteration] == list(range(5001, iteration+1))
    checks, drift = {}, []
    for name, digest in json.loads((RUN/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            checks[name] = frozen.sha(RUN/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/')) == digest
            if frozen.sha(ROOT/name) != digest:
                drift.append(name)
        elif name.startswith('resources/'):
            checks[name] = frozen.sha(ROOT/name) == digest
    assert all(checks.values())
    out.mkdir(parents=True, exist_ok=False)
    (out/'checkpoint').mkdir()
    shutil.copy2(source, out/'checkpoint'/source.name)
    frozen.write(out/'checkpoint-audit.json', dict(passed=True, iteration=iteration, source_checkpoint=str(source),
        checkpoint_sha256=frozen.sha(source), snapshot_and_resources=checks, current_workspace_drift=drift,
        speed_limit=saved['speed_limit'], terrain=saved['terrain'], optimizer_updates=0,
        original_training_not_interrupted=True, training_finished=(RUN/'training_report.json').exists()))
    shutil.copy2(__file__, out/'inspection-source.py')


def conditions():
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import from_record
    b = json.loads((EXPERIMENT/'budget.json').read_text())
    return replace(from_record(b['disturbances']), pushes=False,
                   external_force_range=(0., 0.), external_torque_range=(0., 0.))


def nominal(out, iteration):
    from scripts.agent.fudan_ppo_walk import inspect_pd200_training as inspect
    from scripts.agent.motion.acceptance import canonical, compare_baseline
    inspect.MODEL = out/'checkpoint'/f'model_{iteration}.pt'
    inspect.RUN, inspect.EXPERIMENT = RUN, EXPERIMENT
    proof = json.loads((out/'checkpoint-audit.json').read_text())
    inspect.run(out/'nominal', checkpoint_audit=lambda _: proof['snapshot_and_resources'], conditions_override=conditions())
    result = canonical(out/'nominal/gpu', 'fudan_ppo_walk')
    old = json.loads((BASE/'nominal/acceptance/acceptance.json').read_text())
    compare_baseline(result, old)
    target = out/'nominal/acceptance'
    target.mkdir()
    frozen.write(target/'acceptance.json', result)
    inspect.report(out/'nominal')


def extras(out, iteration):
    import torch
    from scripts.agent.fudan_ppo_walk import evaluate
    from scripts.agent.motion.acceptance import canonical, compare_baseline
    torch.set_num_threads(1)
    yaw = []
    for h in (.22, .26, .30):
        for w in (-10., -5., -3., 3., 5., 10.):
            case = dict(id=f'high_yaw_{w:g}_{h:g}', category='spin', duration=7.,
                        segments=[dict(start=0., duration=7., command=[0., w, h])])
            yaw.append(dict(id=case['id']+'__50001', case=case, seed=50001, randomized=False))
    speed = []
    for h in (.22, .26, .30):
        for direction in (-1, 1):
            case = dict(id=f'forward_compare_{direction}_{h:g}', category='forward' if direction > 0 else 'backward',
                duration=7., segments=[dict(start=0., duration=7., command=[direction*3., 0., h])])
            speed.append(dict(id=f'vx{direction*3:g}_h{h:g}__50001', case=case, seed=50001, randomized=False))
    for label, samples in (('high-yaw', yaw), ('speed3', speed)):
        path = out/label
        request = SimpleNamespace(checkpoint=out/'checkpoint'/f'model_{iteration}.pt', output=path,
            deadline=time.time()+900, diagnostic=True, diagnostic_height=None, batch_size=len(samples),
            selection=None, leg_load_profile='legacy', response_parameters=None)
        with patch.object(evaluate, 'evaluation_items', return_value=samples), \
             patch.object(evaluate, 'from_sim2sim_config', return_value=conditions()):
            evaluate.gpu(request)
        result = canonical(path, 'fudan_ppo_walk')
        if label == 'high-yaw':
            baseline = json.loads((BASE/'high-yaw/5000/acceptance/acceptance.json').read_text())
            if baseline['cohort_sha256'] == result['cohort_sha256']:
                compare_baseline(result, baseline)
        frozen.write(path/'acceptance.json', result)


def routes():
    result = []
    for k, (cx, cy) in enumerate(((4., 4.), (12., 4.), (4., 12.), (12., 12.))):
        angle = 10.+k*10./3
        for mode in ('up', 'down'):
            for speed in (.5, 1.):
                drive = 4./speed
                start = (cx-4., cy) if mode == 'up' else (cx, cy)
                target = (cx, cy) if mode == 'up' else (cx-4., cy)
                result.append(dict(id=f'pyramid_{k}_{mode}_v{speed:g}', family=mode, slope_deg=angle,
                    start_xy=start, target_xy=target, command_vx=speed if mode == 'up' else -speed,
                    drive_s=drive, duration=drive+3., height=.26))
    for k, (cx, cy) in enumerate(((4., -12.), (12., -12.), (4., -4.), (12., -4.))):
        for direction in (-1, 1):
            for speed in (.5, 1.):
                result.append(dict(id=f'rough_{k}_dir{direction}_v{speed:g}', family='rough', slope_deg=0.,
                    start_xy=(cx-direction*2., cy), target_xy=(cx+direction*2., cy),
                    command_vx=direction*speed, drive_s=4./speed, duration=4./speed+3., height=.26))
    return result


def body_contact_allowed(sample):
    # 仅上下坡路线允许擦地；崎岖路线与原47平地的接触要求保留。
    return sample['family'] in ('up', 'down')


def route_failure(sample, substeps):
    if abs(substeps[:, 12:14]).max() > 60.:
        return 'tilt_over60'
    if not body_contact_allowed(sample) and substeps[:, 16].max() > 10.:
        return 'body_contact'
    return ''


def terrain(out, iteration, height_sweep=False, models=None):
    import mujoco
    import numpy as np
    import torch
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    from ct_mjlab.fudan_ppo_walk.measurement import configure_samples, update_evaluation_commands, gpu_row, arrays
    from scripts.agent.motion.assess import phase_verdict
    samples = routes()
    if height_sweep:
        samples = [dict(s, id=s['id']+f'_h{height:g}', height=height)
                   for height in (.22, .30) for s in samples if s['family'] != 'rough']
    protocol = dict(version='regional-route-diagnostic-slope-contact-allowed-v2', samples=samples,
        heights_m=[.22, .30] if height_sweep else [.26], drive_s='4m / requested speed, followed by3s zero command',
        success='No fall; endpoint within0.65m and cross-track<=0.5m; drive2s-to-stop uses common vx/yawMAE<=0.1 and pitch<=5deg; final2s speed<=0.05m/s and pitch<=5deg; both-wheel20ms support; body contact allowed on up/down routes, rough contact<=1N',
        body_contact_policy=dict(allowed_route_families=['up', 'down'],
            scope='Entire declared pyramid up/down route, including its approach and final platform',
            physical_collision_enabled=True,force_recorded=True,
            no_contact_failure_on_slopes=True,rough_and_flat_acceptance_unchanged=True),
        not_navigation=True, optimizer_updates=0, validation_backend='mujoco_warp_gpu')
    folder = 'terrain-height-extra' if height_sweep else 'terrain'
    plan = out/(folder+'-protocol.json')
    if not plan.exists():
        frozen.write(plan, protocol)
    else:
        assert json.loads(plan.read_text()) == json.loads(json.dumps(protocol))
    b = json.loads((EXPERIMENT/'budget.json').read_text())
    torch.set_num_threads(1)
    for label, checkpoint in (models if models is not None else
                             (('5000', SOURCE), (str(iteration), out/'checkpoint'/f'model_{iteration}.pt'))):
        path = out/folder/label
        path.mkdir(parents=True, exist_ok=False)
        fixture, trace_dir = path/'fixtures', path/'traces'
        trace_dir.mkdir()
        policy, meta = load_policy(checkpoint, 'cuda:0')
        cfg = FudanWalkCfg()
        for name, values in b['actual_config'].items():
            for key, value in values.items():
                setattr(getattr(cfg, name), key, value)
        env = FudanEnv(len(samples), seed=50001, randomized=True, cfg=cfg, disturbances=conditions())
        try:
            items = [dict(id=s['id'], case=dict(id=s['id'], category='forward', duration=s['duration'],
                segments=[dict(start=0., duration=s['drive_s'], command=[s['command_vx'], 0., s['height']]),
                          dict(start=s['drive_s'], duration=3., command=[0., 0., s['height']])]),
                seed=50001, randomized=False) for s in samples]
            obs, _ = configure_samples(env, items, fixture)
            sim = env.env.sim
            data = mujoco.MjData(sim.mj_model)
            mujoco.mj_forward(sim.mj_model, data)
            heights = []
            for s in samples:
                x, y = s['start_xy']
                distance = mujoco.mj_ray(sim.mj_model, data, np.array([x, y, 3.]), np.array([0., 0., -1.]),
                    np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, None)
                assert distance >= 0.
                heights.append(3.-distance)
            # 在实际坡脚/平顶/崎岖上方留3cm几何净空，首拍前放置；之后仅由GPU物理推进。
            xy = torch.tensor([s['start_xy'] for s in samples], device=env.device)
            sim.data.qpos[:, :2] = xy
            sim.data.qpos[:, 2] = env.action.reset_root_height+torch.tensor(heights, device=env.device)+.03
            sim.data.qvel.zero_()
            env.env.scene.terrain.env_origins[:, :2] = xy
            env.env.scene.terrain.env_origins[:, 2] = torch.tensor(heights, device=env.device)
            sim.forward(); env.env.scene.update(dt=0.); sim.sense()
            contacts = sim.wp_data
            ncon = int(contacts.nacon.numpy()[0])
            pairs = contacts.contact.geom.numpy()[:ncon]
            terrain_ids = np.flatnonzero(sim.mj_model.geom_bodyid == sim.mj_model.body('terrain').id)
            mask = np.isin(pairs, terrain_ids).any(-1)
            distances = contacts.contact.dist.numpy()[:ncon][mask]
            initial_penetration = float(distances.min()) if len(distances) else 0.
            assert initial_penetration >= -.0001, initial_penetration
            env.previous_position.copy_(env.robot.data.root_link_pos_w)
            env._update_state()
            obs, _ = env.observations()
            env.history.copy_(obs.repeat(1, 5))
            initial = sim.data.qpos.clone()
            np.savez_compressed(path/'initial.npz', qpos=initial.cpu().numpy(), qvel=sim.data.qvel.cpu().numpy(),
                                origins=env.env.scene.env_origins.cpu().numpy())
            env.action.record_substeps = True
            rows = [gpu_row(env.env).cpu().numpy()]
            ground = [env.height.cpu().numpy().copy()]
            poses = [sim.data.qpos.cpu().numpy().copy()]
            failed = np.zeros(len(samples), dtype=bool)
            fail_reason = ['' for _ in samples]
            fail_tick = np.full(len(samples), -1, dtype=int)
            count = round(max(s['duration'] for s in samples)/env.dt)
            for tick in range(count):
                commands = [[s['command_vx'] if tick*env.dt < s['drive_s'] else 0., 0., s['height']] for s in samples]
                # 局部复位也会写传感器状态；no_grad保留普通张量，避免跨InferenceMode写入限制。
                with torch.no_grad():
                    obs = update_evaluation_commands(env, commands, obs)
                    actions, _ = policy.act_inference(obs, env.history)
                    obs, _, _, done, _, _ = env.step(actions, auto_reset=False, resample=False)
                sub = torch.stack(env.action.substep_rows).cpu().numpy()
                rows.extend(sub)
                ground.append(env.height.cpu().numpy().copy())
                poses.append(sim.data.qpos.cpu().numpy().copy())
                for i, s in enumerate(samples):
                    if tick*env.dt >= s['duration'] or failed[i]:
                        continue
                    reason = route_failure(s, sub[:, i])
                    bad = bool(reason or env.terrain_invalid[i].item() or done[i].item())
                    if bad:
                        failed[i] = True
                        fail_reason[i] = reason or ('terrain_invalid' if env.terrain_invalid[i].item() else 'task_termination')
                        fail_tick[i] = tick+1
                inactive = np.array([failed[i] or (tick+1)*env.dt >= s['duration'] for i, s in enumerate(samples)])
                # 已结束世界回安全出生点，后续状态不进入其轨迹或得分，防止倒地后数值发散影响同批其他工况。
                ids = torch.as_tensor(np.flatnonzero(inactive), device=env.device)
                if len(ids):
                    env.reset_idx(ids)
                if tick%200 == 0:
                    print(json.dumps(dict(terrain_model=label, step=tick, total=count, failures=int(failed.sum()))), flush=True)
            rows, poses, ground = np.asarray(rows), np.asarray(poses), np.asarray(ground)
            scores = []
            for i, s in enumerate(samples):
                end_steps = min(round(s['duration']/env.dt), fail_tick[i] if fail_tick[i] >= 0 else count)
                n = end_steps*cfg.control.decimation+1
                trace = arrays(rows[:n, i], poses[:end_steps+1, i], physics_dt=cfg.sim.dt, policy_dt=env.dt)
                trace['terminated'][-1] = failed[i]
                trace['position_world'] = trace['qpos'][:, :3].copy()
                trace['height_local_policy'] = ground[:end_steps+1, i]
                np.savez_compressed(trace_dir/(s['id']+'.npz'), **trace)
                t = trace['time']
                stop = (t >= s['duration']-2.) & (t < s['duration'])
                drive = (t >= 2.) & (t < s['drive_s'])
                complete = not failed[i] and end_steps == round(s['duration']/env.dt)
                speed = np.linalg.norm(trace['linear_world'][:, :2], axis=-1)
                def phase(mask, target, parking):
                    return phase_verdict(vx=trace['linear_body'][:, 0], yaw=trace['angular_body'][:, 2],
                        horizontal_speed=speed, pitch_deg=trace['roll_pitch_deg'][:, 1], vx_target=target, yaw_target=0.,
                        selected=mask, valid=np.ones(len(t), bool), complete=complete, parking=parking)
                motion, stopping = phase(drive, s['command_vx'], False), phase(stop, 0., True)
                final = trace['qpos'][-1, :2]
                target = np.array(s['target_xy'])
                endpoint = float(np.linalg.norm(final-target))
                cross = float(abs(final[1]-target[1]))
                reached = bool(complete and endpoint <= .65 and cross <= .5)
                body_ok = bool(stop.any() and trace['base_contact'][stop].max() <= 1.)
                contact_check_applied = not body_contact_allowed(s)
                # 以20ms窗口内平均支撑判断双轮承重；原始1ms轮载同时保留，不把单个零接触拍当通行成功。
                supports = trace['wheel_support'][stop]
                window = round(.02/cfg.sim.dt)
                cumulative = np.vstack((np.zeros((1, 2)), supports.cumsum(0)))
                smoothed = (cumulative[window:]-cumulative[:-window])/window
                supported = bool(len(smoothed) and (smoothed >= 5.).all())
                passed = reached and motion['passed'] and stopping['passed'] and (body_ok or not contact_check_applied) and supported
                contact_ticks = np.flatnonzero(trace['base_contact'] > 10.)
                scores.append(dict(**s, passed=passed, complete=complete, fell=fail_reason[i]=='tilt_over60',
                    failure_reason=fail_reason[i],
                    reached=reached, endpoint_error_m=endpoint, cross_track_error_m=cross,
                    final_xy_m=final.tolist(), mean_drive_vx_m_s=float(trace['linear_body'][drive, 0].mean()) if drive.any() else None,
                    drive=motion, stop=stopping, body_contact_ok=body_ok,
                    body_contact_check_applied=contact_check_applied,
                    body_contact_peak_N=float(trace['base_contact'].max()),
                    first_body_contact_over10_s=float(t[contact_ticks[0]]) if len(contact_ticks) else None,
                    both_wheel_20ms_support_ok=supported))
            frozen.write(path/'result.json', dict(rows=scores, passed=sum(s['passed'] for s in scores), total=len(scores),
                checkpoint=str(checkpoint), checkpoint_sha256=frozen.sha(checkpoint), optimizer_updates=0,
                physics_steps=count*cfg.control.decimation, no_parameter_updates=True,
                initial_minimum_contact_distance_m=initial_penetration))
        finally:
            env.close()


def report(out, iteration):
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    current = json.loads((out/'nominal/acceptance/acceptance.json').read_text())
    old = json.loads((BASE/'nominal/acceptance/acceptance.json').read_text())
    terrain_results = {label: json.loads((out/'terrain'/label/'result.json').read_text()) for label in ('5000', str(iteration))}
    for label, result in terrain_results.items():
        extra = out/'terrain-height-extra'/label/'result.json'
        if extra.exists():
            result['rows'].extend(json.loads(extra.read_text())['rows'])
    summaries = {}
    for label, result in terrain_results.items():
        summaries[label] = {family: dict(passed=sum(r['passed'] for r in result['rows'] if r['family'] == family),
            total=sum(r['family'] == family for r in result['rows']), reached=sum(r['reached'] for r in result['rows'] if r['family'] == family),
            fell=sum(r['fell'] for r in result['rows'] if r['family'] == family)) for family in ('up', 'down', 'rough')}
    events = EventAccumulator(str(RUN), size_guidance={'scalars': 0}).Reload()
    rewards = [v.value for v in events.Scalars('Train/reward') if v.step <= iteration]
    result = dict(iteration=iteration, nominal_before=old['technical_passed'], nominal_after=current['technical_passed'],
        nominal_total=47, nominal_comparison=current.get('comparison'), terrain=summaries,
        reward_first100=float(np.mean(rewards[:100])), reward_last100=float(np.mean(rewards[-100:])),
        source_model_unchanged=frozen.sha(SOURCE) == '010106ac0b2ee356a630ab27ebe1c85ab2a6c843d21a1b4362ab81f1aafe0222',
        original_training_not_interrupted=True, optimizer_updates=0, behavior_accepted=False)
    for label in ('high-yaw', 'speed3'):
        path = out/label/'acceptance.json'
        if path.exists():
            d = json.loads(path.read_text())
            result[label] = dict(passed=d['technical_passed'], total=d['total'])
    frozen.write(out/'comparison.json', result)
    lines = [f'# 分区训练{iteration}阶段效果', '', f'冻结{iteration}完整模型，原训练继续按8000预算运行；本次不追加训练。', '',
        f'原平地47：5000的{old["technical_passed"]}/47 → 当前{current["technical_passed"]}/47。', '',
        '| 模型 | 上坡＋停车 | 下坡＋停车 | 崎岖通行＋停车 |', '|---|---|---|---|']
    for label, row in summaries.items():
        lines.append('| '+label+' | '+' | '.join(f'{row[k]["passed"]}/{row[k]["total"]}（到目标{row[k]["reached"]}，失败/摔倒{row[k]["fell"]}）' for k in ('up','down','rough'))+' |')
    lines += ['', '地形为名义物理、无外载/推扰，坡道三高度0.22/0.26/0.30m、崎岖0.26m，速度0.5/1m/s，给定方向行驶4m后停车3秒；路线与门槛在各组采集前保存。不是导航、全随机分布、坡上3m/s或完整任务验收。原平地验收门槛不变，新增路线分开报告。', '',
        f'训练首/末100平均reward：{result["reward_first100"]:.6f} / {result["reward_last100"]:.6f}；不能替代通行与平地保留。', '',
        '详细逐例原因见nominal/acceptance/acceptance.json与terrain/*/result.json。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def render(out, iteration, *, labels=None, chosen=None, output_dir=None):
    from contextlib import ExitStack
    import hashlib
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    output = out/'videos' if output_dir is None else output_dir
    output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 19)
    title = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 23)
    if chosen is None:
        chosen = ['pyramid_0_up_v1', 'pyramid_3_up_v1_h0.3', 'pyramid_3_down_v0.5_h0.3', 'rough_0_dir1_v1']
    if labels is None:
        labels = ('5000', str(iteration))
    index = []
    with ExitStack() as stack:
        sides = []
        for label in labels:
            root = out/'terrain'/label
            scores = {r['id']: r for r in json.loads((root/'result.json').read_text())['rows']}
            extra_root = out/'terrain-height-extra'/label
            scores.update({r['id']: r for r in json.loads((extra_root/'result.json').read_text())['rows']})
            model = mujoco.MjModel.from_binary_path(str(root/'fixtures/model.mjb'))
            disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth, model.vis.global_.offheight = 480, 400
            model.vis.headlight.ambient[:] = .2
            model.vis.headlight.diffuse[:] = .25
            model.light_diffuse[:] = .2
            data = mujoco.MjData(model)
            renderer = stack.enter_context(mujoco.Renderer(model, width=480, height=400))
            camera = mujoco.MjvCamera()
            camera.distance, camera.elevation, camera.azimuth = 1.6, -13., 90.
            option = mujoco.MjvOption()
            option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
            sides.append((label, root, scores, model, data, renderer, camera, option))
        for sid in chosen:
            traces, scores = [], []
            for label, root, lookup, *_ in sides:
                scores.append(lookup[sid])
                trace_root = out/'terrain-height-extra'/label if '_h' in sid else root
                with np.load(trace_root/'traces'/(sid+'.npz')) as saved:
                    traces.append({k: saved[k].copy() for k in saved.files})
            duration = scores[0]['duration']
            path = output/(sid+'.mp4')
            frames = round(duration*50)
            with imageio.get_writer(path, fps=50, codec='libx264', quality=8) as writer:
                for frame in range(frames):
                    t = frame/50
                    canvas = Image.new('RGB', (960, 576), '#18212b')
                    draw = ImageDraw.Draw(canvas)
                    mode = '上坡后停车' if scores[0]['family'] == 'up' else ('下坡后停车' if scores[0]['family'] == 'down' else '崎岖通行后停车')
                    draw.text((14, 6), f'正常1倍速 | {mode} | {scores[0]["slope_deg"]:g}° | vx={scores[0]["command_vx"]:g}m/s | h={scores[0]["height"]:g}m', font=title, fill='white')
                    for column, (side, trace, score) in enumerate(zip(sides, traces, scores)):
                        label, _, _, model, data, renderer, camera, option = side
                        qi = min(int(round(t/.01)), len(trace['qpos'])-1)
                        tick = min(int(round(t/.001)), len(trace['time'])-1)
                        data.qpos[:] = trace['qpos'][qi]
                        mujoco.mj_forward(model, data)
                        camera.lookat[:] = data.qpos[:3]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        canvas.paste(Image.fromarray(renderer.render().copy()), (column*480, 144))
                        x = column*480+12
                        draw = ImageDraw.Draw(canvas)
                        draw.text((x, 48), f'{label}轮  通行＋停车：'+('通过' if score['passed'] else '未通过'), font=title,
                            fill='#a9edb0' if score['passed'] else '#ff8e8e')
                        phase = '行驶' if t < score['drive_s'] else '停车'
                        draw.text((x, 82), f'{phase} t={t:.2f}s  实测vx={trace["linear_body"][tick, 0]:+.2f}m/s', font=font, fill='white')
                        draw.text((x, 111), f'终点误差{score["endpoint_error_m"]:.2f}m；'+('已结束，后段停帧' if t > trace['qpos_time'][-1]+.01 else '保存GPU状态回放'), font=font, fill='white')
                    draw.text((14, 545), '相同路线与名义物理；失败后的面板停帧，停止状态不计为停车成功', font=font, fill='white')
                    if frame == min(frames-1, 250):
                        canvas.save(path.with_suffix('.png'))
                    writer.append_data(np.asarray(canvas))
            with imageio.get_reader(path) as reader:
                assert reader.get_meta_data()['fps'] == 50
                assert sum(1 for _ in reader) == frames
            index.append(dict(sample_id=sid, duration_s=duration, fps=50, frames=frames, playback_speed=1,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), full_decode_verified=True,
                sources=[dict(label=s[0], trajectory=str((out/'terrain-height-extra'/s[0] if '_h' in sid else s[1])/'traces'/(sid+'.npz')),
                    trajectory_sha256=frozen.sha((out/'terrain-height-extra'/s[0] if '_h' in sid else s[1])/'traces'/(sid+'.npz'))) for s in sides]))
    frozen.write(output/'video-index.json', dict(videos=index, physics_steps=0, optimizer_updates=0))


def plot(out, iteration):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(RUN), size_guidance={'scalars': 0}).Reload()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for axis, tag in zip(axes, ('Train/reward', 'Train/speed_limit')):
        values = [v for v in events.Scalars(tag) if v.step <= iteration]
        axis.plot([v.step for v in values], [v.value for v in values], lw=.6)
        axis.set(ylabel=tag)
        axis.grid(alpha=.25)
    axes[-1].set_xlabel('Iteration')
    fig.tight_layout(); fig.savefig(out/'reward-and-curriculum.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    cases = {r['id']: r for r in routes()}
    for row, mode in enumerate(('up', 'down')):
        for k in range(4):
            sid = f'pyramid_{k}_{mode}_v1'
            case = cases[sid]
            axis = axes[row, k]
            for label in ('5000', str(iteration)):
                with np.load(out/'terrain'/label/'traces'/(sid+'.npz')) as data:
                    direction = 1 if mode == 'up' else -1
                    progress = (data['qpos'][:, 0]-case['start_xy'][0])*direction
                    axis.plot(data['qpos_time'], progress, label=label)
            axis.axhline(4., color='black', ls='--', lw=.8)
            axis.axvline(case['drive_s'], color='grey', ls=':', lw=.8)
            axis.set(title=f'{mode} / {case["slope_deg"]:.2f} deg', xlabel='Time (s)', ylabel='Progress (m)')
            axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(out/'terrain-progress.png', dpi=150); plt.close(fig)


def finalize(out, iteration):
    import numpy as np
    report(out, iteration)
    compared, differences = 0, []
    by_height = {}
    for folder in ('terrain', 'terrain-height-extra'):
        before, after = out/folder/'5000', out/folder/str(iteration)
        for path in sorted((before/'fixtures').glob('*_initial.npz'))+[before/'initial.npz']:
            relative = path.relative_to(before)
            with np.load(path) as left, np.load(after/relative) as right:
                assert set(left.files) == set(right.files)
                for key in left.files:
                    compared += 1
                    if not np.array_equal(left[key], right[key]):
                        differences.append(dict(path=str(relative), field=key,
                            max_difference=float(np.abs(left[key]-right[key]).max())))
        for label in ('5000', str(iteration)):
            result = json.loads((out/folder/label/'result.json').read_text())
            assert result['initial_minimum_contact_distance_m'] >= -.0001
            for row in result['rows']:
                with np.load(out/folder/label/'traces'/(row['id']+'.npz')) as trace:
                    assert all(np.isfinite(trace[k]).all() for k in trace.files)
                    assert abs(trace['time'][-1]-trace['qpos_time'][-1]) < 1e-8
                    if row['complete']:
                        assert abs(trace['time'][-1]-row['duration']) < 1e-8
                    else:
                        assert trace['terminated'][-1]
                key = f'{label}_h{row["height"]:g}_{row["family"]}'
                group = by_height.setdefault(key, dict(total=0, reached=0, passed=0, fell=0))
                group['total'] += 1
                for name in ('reached', 'passed', 'fell'):
                    group[name] += int(row[name])
    frozen.write(out/'terrain-pairing-audit.json', dict(initial_arrays_compared=compared,
        all_initial_arrays_bitwise_equal=not differences, differences=differences,
        full_finite_traces_verified=True, by_height=by_height, optimizer_updates=0))
    shutil.copy2(__file__, out/'inspection-final-source.py')
    with (out/'RESULTS.md').open('a') as stream:
        stream.write('\n## 补充核验与视频\n\n')
        stream.write(f'两模型原始初始化数组逐字段比较{compared}项，严格逐值相同：{not differences}；完整轨迹有限、结束时刻与失败截断已核。不能把失败后的安全复位状态计入成功。\n\n')
        stream.write('首两次地形采集因InferenceMode状态写入限制失败，保留于terrain-attempt01/02，未计作策略失败；有效采集使用no_grad，0学习。\n\n')
        stream.write('[路线进度曲线](terrain-progress.png) · [训练奖励和速度课程](reward-and-curriculum.png)\n\n')
        for sid, label in [('pyramid_0_up_v1', '10°上坡，0.26m高'),
                           ('pyramid_3_up_v1_h0.3', '20°上坡，0.30m高'),
                           ('pyramid_3_down_v0.5_h0.3', '20°低速下坡，0.30m高'),
                           ('rough_0_dir1_v1', '小崎岖1m/s通行')]:
            stream.write(f'- [{label}，左5000／右{iteration}，正常1倍速](videos/{sid}.mp4)\n')
    print(json.dumps(dict(terrain_by_height=by_height, paired_arrays=compared, initial_differences=differences)), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('audit', 'nominal', 'extras', 'terrain', 'height-sweep', 'report', 'render', 'plot', 'finalize'))
    p.add_argument('--iteration', type=int, default=7600)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if a.mode == 'audit':
        audit(out, a.iteration)
        return
    proof = json.loads((out/'checkpoint-audit.json').read_text())
    assert proof['passed'] and proof['iteration'] == a.iteration
    assert frozen.sha(out/'checkpoint'/f'model_{a.iteration}.pt') == proof['checkpoint_sha256']
    frozen.RUN = RUN
    sys.meta_path.insert(0, frozen.FrozenTrainingSource())
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import fudan_ppo_walk as physical
    original = physical.make_env_cfg
    def configure(*args, **kwargs):
        cfg = original(*args, **kwargs)
        cfg.sim.nconmax, cfg.sim.njmax = 64, 256
        return cfg
    with patch.object(physical, 'make_env_cfg', configure):
        if a.mode == 'height-sweep':
            terrain(out, a.iteration, True)
        else:
            globals()[a.mode](out, a.iteration)
    frozen.write(out/(a.mode+'-frozen-execution.json'), dict(modules=frozen.LOADED, optimizer_updates=0))


if __name__ == '__main__':
    main()
