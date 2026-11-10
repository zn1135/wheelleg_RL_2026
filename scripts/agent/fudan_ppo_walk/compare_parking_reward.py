"""Bounded GPU comparison of frozen 8000/9000 policies, using archived identical inputs."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

HEIGHTS = ('h022', 'h026', 'h030')
BRANCHES = ('baseline', 'candidate')
PARENT = Path('logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt')
CANDIDATE = Path('logs/mjlab/fudan_ppo_walk/20261103_parking_reward_8000_to9000/model_final.pt')
RECORDING = Path('artifacts/fudan_ppo_walk/fudan-gpu-replay-8000-20261102')
MODEL_HASHES = ('39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9',
                '549048b79b95d882f65f722beb909eb62cf47caee1aff31d7dd16215429c8692')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def guard(root, replay=False):
    budget = json.loads((root / 'budget.json').read_text())
    if time.time() >= budget['start_unix'] + (345 if replay else 590):
        raise TimeoutError('Stop before the authorized deadline; no retry')


def replay(root):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    from ct_mjlab.fudan_ppo_walk.measurement import gpu_row, arrays, update_evaluation_commands
    from .diagnose_parking_integral import restore_initial
    from .evaluate import command
    torch.set_num_threads(1)
    assert not FudanWalkCfg().parking_integral.enabled
    disturbances = from_task_config()
    assert not (disturbances.action_delay or disturbances.sensor_delay or disturbances.pushes)
    policies = [load_policy(p, 'cuda:0') for p in (PARENT, CANDIDATE)]
    assert [m['iterations'] for _, m in policies] == [8000, 9000]
    assert policies[0][1]['reset_contract'] == policies[1][1]['reset_contract']
    frozen = [{k: v.detach().cpu().clone() for k, v in p.state_dict().items()} for p, _ in policies]
    checks = []; prepared = []; executions = []
    def forbidden(*args, **kwargs):
        raise AssertionError('Optimizer updates forbidden during evaluation')
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam, 'step', side_effect=forbidden))
        for height in HEIGHTS:
            guard(root, True)
            source = RECORDING / height / 'gpu15'
            protocol = json.loads((source / 'protocol.json').read_text())
            items = protocol['samples']; assert len(items) == 15
            folders = {}
            for branch in BRANCHES:
                folder = root / height / branch; folder.mkdir(parents=True)
                (folder / 'fixtures').mkdir(); (folder / 'traces').mkdir()
                write(folder / 'protocol.json', dict(protocol, mode='diagnostic_not_formal_acceptance',
                    validation_engine='gpu', validation_backend='mujoco_warp_gpu', integral_enabled=False))
                folders[branch] = folder
            env = FudanEnv(30, seed=50001, randomized=True, reset_contract=policies[0][1]['reset_contract'], disturbances=disturbances)
            stack.callback(env.close)
            try:
                obs, _ = restore_initial(env, items, source / 'fixtures', folders, checks, height, branches=BRANCHES)
            finally:
                write(root / 'initial-consistency.json', checks)
            prepared.append((height, items, env, obs))
        assert len(checks) == 90 and all(c['all_arrays_identical'] for c in checks)
        write(root / 'initialization-result.json', dict(passed=True, checked=90, policy_batches_before_checks=0))
        for height, items, env, obs in prepared:
            guard(root, True); env.action.record_substeps = True
            rows = [gpu_row(env.env).cpu().numpy()]
            poses = [env.env.sim.data.qpos.cpu().numpy().copy()]
            terminated = [np.zeros(30, dtype=bool)]
            outputs = []; estimates = []; truth = []; completed = 0; error = None
            try:
                with torch.inference_mode():
                    for tick in range(1500):
                        guard(root, True)
                        raw = obs.new_tensor([command(s['case'], tick * .01) for s in items] * 2)
                        obs = update_evaluation_commands(env, raw, obs)
                        actions = []; encoded = []
                        for b, (policy, _) in enumerate(policies):
                            lo, hi = b * 15, (b + 1) * 15
                            action, estimate = policy.act_inference(obs[lo:hi], env.history[lo:hi])
                            actions.append(action); encoded.append(estimate)
                        action = torch.cat(actions); estimate = torch.cat(encoded)
                        assert torch.isfinite(action).all() and torch.isfinite(estimate).all()
                        outputs.append(action.cpu().numpy()); estimates.append((estimate / 2.).cpu().numpy())
                        truth.append(env.linear.cpu().numpy().copy())
                        obs, _, _, _, _, _ = env.step(action, auto_reset=False, resample=False)
                        rows.extend(torch.stack(env.action.substep_rows).cpu().numpy())
                        poses.append(env.env.sim.data.qpos.cpu().numpy().copy())
                        terminated.extend([(env.fail_ticks > 100).cpu().numpy().copy()] * 5)
                        completed = tick + 1
                        if tick % 300 == 0:
                            print(json.dumps(dict(height=height, step=completed, total=1500)), flush=True)
            except Exception as exc:
                error = repr(exc)
            finally:
                rows = np.asarray(rows); poses = np.asarray(poses); terminated = np.asarray(terminated)
                origins = env.env.scene.env_origins.cpu().numpy()
                for b, branch in enumerate(BRANCHES):
                    folder = root / height / branch
                    for i, sample in enumerate(items):
                        idx = b * 15 + i; duration = sample['case']['duration']
                        end = min(len(rows), round(duration / .002) + 1)
                        count = min(completed, round(duration / .01))
                        q = poses[:min(len(poses), count + 1), idx].copy(); q[:, :3] -= origins[idx]
                        trace = arrays(rows[:end, idx], q, terminated=terminated[:end, idx])
                        if outputs:
                            trace.update(action=np.asarray(outputs)[:count, idx], encoder_m_s=np.asarray(estimates)[:count, idx],
                                policy_truth_m_s=np.asarray(truth)[:count, idx], policy_time=np.arange(count) * .01)
                        np.savez_compressed(folder / 'traces' / (sample['id'] + '.npz'), **trace)
                    write(folder / 'execution.json', dict(engine='mjlab', validation_backend='mujoco_warp_gpu',
                        complete_traces=sum(completed * .01 >= s['case']['duration'] for s in items), planned=15,
                        policy_batches=completed, integral_enabled=False, error=error, optimizer_updates=0))
                executions.append(dict(height=height, policy_steps=completed, error=error))
                write(root / 'replay-executions.json', executions)
            if error:
                raise RuntimeError(error)
    assert all(torch.equal(f[k], v.detach().cpu()) for f, (p, _) in zip(frozen, policies) for k, v in p.state_dict().items())
    write(root / 'frozen-policies.json', dict(identical=True, optimizer_updates=0,
        policy_batches=9000, environment_physics_substeps=675000, integral_enabled=False))


def report(root):
    from scripts.agent.motion.acceptance import canonical, compare_baseline
    from .report_parking_integral import parking_metrics
    rows = []; summaries = {b: dict(passed=0, total=45, regressed=[], gained=[]) for b in BRANCHES}
    for height in HEIGHTS:
        guard(root); reports = {}
        for branch in BRANCHES:
            folder = root / height / branch
            reports[branch] = canonical(folder, 'fudan_ppo_walk')
        reports['candidate'] = compare_baseline(reports['candidate'], reports['baseline'])
        archived = canonical(RECORDING / height / 'gpu15', 'fudan_ppo_walk')
        reports['baseline'] = compare_baseline(reports['baseline'], archived)
        for branch in BRANCHES:
            result = reports[branch]; folder = root / height / branch / 'acceptance'; folder.mkdir()
            write(folder / 'acceptance.json', result)
            summaries[branch]['passed'] += result['technical_passed']
            for key in ('regressed', 'gained'):
                summaries[branch][key].extend(height + '/' + sid for sid in result['comparison'][key])
            for item in result['samples']:
                if item['case']['category'] not in ('zero', 'stop_forward', 'stop_backward'):
                    continue
                sid = item['sample_id']; lo, hi = (2., 7.) if item['case']['category'] == 'zero' else (10., 15.)
                with np.load(root / height / branch / 'traces' / (sid + '.npz')) as trace:
                    metrics = parking_metrics(trace, lo, hi)
                    indices = slice(round(lo / .01), round(hi / .01))
                    metrics['mean_encoder_vx_m_s'] = float(trace['encoder_m_s'][indices, 0].mean())
                    metrics['mean_policy_true_vx_m_s'] = float(trace['policy_truth_m_s'][indices, 0].mean())
                if branch == 'baseline':
                    with np.load(RECORDING / height / 'gpu15/traces' / (sid + '.npz')) as old:
                        previous = parking_metrics(old, lo, hi)
                    metrics['archive_vector_difference_m'] = float(np.linalg.norm(np.array([metrics['net_x_m'], metrics['net_y_m']]) -
                        np.array([previous['net_x_m'], previous['net_y_m']])))
                rows.append(dict(height=height, branch=branch, id=sid, category=item['case']['category'], **metrics))
    reproduced = all(r.get('archive_vector_difference_m', 0) <= .02 for r in rows if r['branch'] == 'baseline')
    result = dict(baseline_reproduced=reproduced, unified=summaries, rows=rows,
        parking_passed={b: sum(r['passed'] for r in rows if r['branch'] == b) for b in BRANCHES},
        fixed_windows=dict(zero=[2,7],stop=[10,15]), strict_parking_thresholds=dict(offset_m=.05,mean_speed_m_s=.01,peak_speed_m_s=.05),
        integral_enabled=False, optimizer_updates=0, final_acceptance='pending_user_video_review')
    write(root / 'comparison.json', result)
    lines = ['# 8000→9000停车奖励GPU对照', '', '三档高度各15工况，名义物理；不是正式987例验收。积分补偿全关。', '',
        '| 高度/工况 | 8000：偏移cm / 均速cm/s | 9000：偏移cm / 均速cm/s |', '|---|---:|---:|']
    for height in HEIGHTS:
        for sid in ('diagnostic_0','diagnostic_13','diagnostic_14'):
            values = [next(r for r in rows if r['height']==height and r['id']==sid and r['branch']==b) for b in BRANCHES]
            lines.append('| '+height+'/'+sid+' | '+' | '.join(f"{r['maximum_horizontal_offset_m']*100:.2f} / {r['mean_horizontal_speed_m_s']*100:.3f}" for r in values)+' |')
    lines += ['', f"严格停车9例：{result['parking_passed']}；统一技术结果：{summaries}。", '',
        '停车固定窗口：原地2～7s，驾驶后停车10～15s；偏移含终点，速度取500Hz世界水平速度模长，不抵消方向。严格停车要求最大偏移≤5cm、均速≤0.01m/s、峰速≤0.05m/s，另列于统一运动评分。', '',
        '统一评分使用共用0.10m/s前向MAE、0.10rad/s转向MAE、停车峰速0.05m/s及pitch峰5°，保留高度/接触等原判据。初态严格恢复归档，不加热步；原模型重跑数值分歧用停车净位移矢量差≤2cm核复现。', '',
        '[详细比较](comparison.json) · [预算](budget.json) · [初态核对](initial-consistency.json)', '',
        '正常速度完整停车视频：[0.22m](videos/h022.mp4) · [0.26m](videos/h026.mp4) · [0.30m](videos/h030.mp4)。最终待用户视频验收。']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k:result[k] for k in ('baseline_reproduced','parking_passed','unified')}, ensure_ascii=False), flush=True)


def video(root):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    (root / 'videos').mkdir(); index = []; review = []
    for height in HEIGHTS:
        guard(root)
        model = mujoco.MjModel.from_binary_path(str(root / height / 'baseline/fixtures/model.mjb'))
        disable_model_sameframe_shortcuts(model); data = mujoco.MjData(model)
        model.vis.global_.offwidth=384; model.vis.global_.offheight=288
        option=mujoco.MjvOption(); option.geomgroup[:]=[1,1,1,0,0,0]
        camera=mujoco.MjvCamera(); camera.distance=1.65; camera.elevation=-18.; camera.azimuth=110.
        frames=0; chapters=[]; path=root/'videos'/(height+'.mp4')
        protocol=json.loads((root/height/'baseline/protocol.json').read_text())
        samples={s['id']:s for s in protocol['samples']}
        with mujoco.Renderer(model,height=288,width=384) as renderer, imageio.get_writer(path,fps=50,codec='libx264',quality=7) as writer:
            for sid in ('diagnostic_0','diagnostic_13','diagnostic_14'):
                traces=[]
                for b in BRANCHES:
                    with np.load(root/height/b/'traces'/(sid+'.npz')) as f:traces.append({k:f[k] for k in f.files})
                expected=round(samples[sid]['case']['duration']*100)+1
                assert all(len(t['qpos'])==expected for t in traces), 'No incomplete video presented as a full replay'
                start=frames
                for tick in range(0,expected-1,2):
                    guard(root); canvas=Image.new('RGB',(768,320),'#101820'); draw=ImageDraw.Draw(canvas)
                    draw.text((8,8),f'8000 LEFT / 9000 RIGHT | {sid} | full startup/braking | normal speed 1x',fill='white')
                    for col,(branch,trace) in enumerate(zip(BRANCHES,traces)):
                        q=trace['qpos'][tick]; data.qpos[:]=q; mujoco.mj_forward(model,data)
                        camera.lookat[:]=q[:3]; camera.lookat[2]-=.025
                        renderer.update_scene(data,camera=camera,scene_option=option)
                        for x in range(int(np.floor(q[0]))-2,int(np.floor(q[0]))+3):
                            if renderer.scene.ngeom>=renderer.scene.maxgeom:break
                            g=renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array([.04,.4,.003]),np.array([float(x),0.,.003]),np.eye(3).reshape(9),np.array([.95,.7,.2,1.]))
                            renderer.scene.ngeom+=1
                        panel=Image.fromarray(renderer.render().copy()); pd=ImageDraw.Draw(panel)
                        pd.rectangle((0,0,384,61),fill='#18212b'); speed=float(np.linalg.norm(trace['linear_world'][tick*5,:2]))
                        from .evaluate import command
                        cmd=command(samples[sid]['case'],tick*.01)
                        pd.text((6,5),f'{branch} | {height} | t={tick*.01:.2f}s | I=OFF',fill='white')
                        pd.text((6,23),f'cmd vx={cmd[0]:+.2f} yaw={cmd[1]:+.2f} | speed={speed:.3f}m/s',fill='white')
                        pd.text((6,41),f'X={q[0]:+.3f}m | pitch={trace["roll_pitch_deg"][tick*5,1]:+.2f}deg',fill='white')
                        canvas.paste(panel,(384*col,32))
                    writer.append_data(np.asarray(canvas)); frames+=1
                    if tick==expected-3: review.append(canvas.copy())
                chapters.append(dict(sample_id=sid,start_s=start/50,duration_s=(frames-start)/50,complete=True))
        guard(root); reader=imageio.get_reader(path); count=0
        for frame in reader:count+=1
        meta=reader.get_meta_data();reader.close()
        assert count==frames==1850 and meta['fps']==50
        index.append(dict(height=height,path=str(path),frames=frames,fps=50,duration_s=frames/50,chapters=chapters,sha256=sha(path),all_frames_decoded=True))
        write(root/'video-index.json',index); print(json.dumps(index[-1]),flush=True)
    sheet=Image.new('RGB',(768*3,320*3),'#101820')
    for i,im in enumerate(review):sheet.paste(im,((i//3)*768,(i%3)*320))
    sheet.save(root/'video-review.png')


def pipeline(root):
    root.mkdir(parents=True,exist_ok=False)
    for p,h in zip((PARENT,CANDIDATE),MODEL_HASHES):assert sha(p)==h
    inputs={str(p):sha(p) for p in RECORDING.glob('h*/gpu15/**/*') if p.is_file()}
    files=[*Path('mjlab_training/src/ct_mjlab').rglob('*.py'),*Path('scripts/agent/fudan_ppo_walk').glob('*.py'),
           *Path('scripts/agent/motion').glob('*.py'),Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')]
    sources={str(p):sha(p) for p in files}
    for p in files:
        target=root/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(root/'source-sha256.json',sources);write(root/'input-sha256.json',inputs)
    started=time.time(); mono=time.monotonic()
    budget=dict(start_unix=started,deadline_unix=started+600,maximum_wall_s=600,replay_maximum_wall_s=360,
        start_local=datetime.fromtimestamp(started,ZoneInfo('Asia/Shanghai')).isoformat(),
        deadline_local=datetime.fromtimestamp(started+600,ZoneInfo('Asia/Shanghai')).isoformat(),
        authorization='用户“做”授权最多600秒GPU三高度45工况及8000/9000停车视频对照；不训练、不调参、不重试',
        model_paths=[str(PARENT),str(CANDIDATE)],model_sha256=MODEL_HASHES,expected_trajectories=90)
    write(root/'budget.json',budget);print(json.dumps(budget,ensure_ascii=False),flush=True)
    env=os.environ.copy();env.update(PYTHONPATH='mjlab_training/src',MUJOCO_GL='egl');steps=[]
    for stage,limit in (('replay',360),('report',600),('video',600)):
        remaining=limit-(time.monotonic()-mono)
        if remaining<=0:break
        with (root/(stage+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-m','scripts.agent.fudan_ppo_walk.compare_parking_reward',stage,'--output',str(root)],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            timed_out=False
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        steps.append(dict(stage=stage,returncode=code,hard_timeout=timed_out,finished_elapsed_s=time.monotonic()-mono))
        write(root/'pipeline-progress.json',steps); print(json.dumps(steps[-1]),flush=True)
        if code:break
    result=dict(steps=steps,wall_s=time.monotonic()-mono,finished_unix=time.time(),
        models_unchanged=all(sha(p)==h for p,h in zip((PARENT,CANDIDATE),MODEL_HASHES)),
        source_unchanged=all(sha(p)==h for p,h in sources.items()),inputs_unchanged=all(sha(p)==h for p,h in inputs.items()),
        optimizer_updates=0,no_retry=True,within_budget=time.monotonic()-mono<=600,
        final_acceptance='pending_user_video_review')
    write(root/'pipeline-result.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=('pipeline','replay','report','video'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    globals()[args.mode](args.output)


if __name__=='__main__':main()
