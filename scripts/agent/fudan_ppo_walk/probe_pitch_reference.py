"""Frozen GPU pitch-reference experiment; actual pitch attainment is measured separately."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
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
from ct_mjlab.fudan_ppo_walk.selected_policy import reference_angles, rotate_gravity, policy_inputs

HEIGHTS = ('h022', 'h026', 'h030')
BRANCHES = ('8000_zero', '8000_plus5', '8000_minus5', '9000_zero', '9000_plus5', '9000_minus5')
TARGETS = (0., 5., -5., 0., 5., -5.)
SIDS = ('diagnostic_0', 'diagnostic_13', 'diagnostic_14')
SOURCE = Path('artifacts/fudan_ppo_walk/fudan-gpu-replay-8000-20261102')
MODELS = (Path('logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt'),
          Path('logs/mjlab/fudan_ppo_walk/20261103_parking_reward_8000_to9000/model_final.pt'))
HASHES = ('39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9',
          '549048b79b95d882f65f722beb909eb62cf47caee1aff31d7dd16215429c8692')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def check(root):
    import torch
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(73)
    obs = torch.randn(6,25); hist = torch.randn(6,125)
    original_obs, original_hist = obs.clone(), hist.clone()
    target = torch.tensor(TARGETS) * np.pi / 180.
    starts = torch.full((6,), 2.)
    before = policy_inputs(obs, hist, 1., starts, target)
    assert torch.equal(before[0],obs) and torch.equal(before[1],hist)
    after, history = policy_inputs(obs, hist, 3., starts, target)
    assert torch.equal(after[[0,3]],obs[[0,3]]) and torch.equal(history[[0,3]],hist[[0,3]])
    unchanged = [i for i in range(25) if i not in (3,5)]
    assert torch.equal(after[:,unchanged],obs[:,unchanged])
    assert torch.equal(history.reshape(6,5,25)[:,:,unchanged],hist.reshape(6,5,25)[:,:,unchanged])
    assert torch.allclose(after[:,3:6].norm(dim=-1),obs[:,3:6].norm(dim=-1),atol=1e-6)
    actual = torch.zeros(6,25); actual[:,3]=target.sin();actual[:,5]=-target.cos()
    corrected = rotate_gravity(actual,target)
    assert torch.allclose(corrected[:,3],torch.zeros(6),atol=1e-7)
    assert torch.allclose(corrected[:,5],-torch.ones(6),atol=1e-7)
    mid = reference_angles(2.25, starts, target)
    assert torch.allclose(mid,target/2)
    assert torch.equal(obs,original_obs) and torch.equal(hist,original_hist)
    for height in HEIGHTS:
        source=SOURCE/height/'gpu15';protocol=json.loads((source/'protocol.json').read_text())
        assert all(any(s['id']==sid for s in protocol['samples']) for sid in SIDS)
        assert all((source/'fixtures'/(sid+'_initial.npz')).is_file() for sid in SIDS)
    assert all(sha(p)==h for p,h in zip(MODELS,HASHES))
    write(root/'checks.json',dict(passed=True,physics_steps=0,model_forwards=0,optimizer_updates=0,
        reference_degrees=list(TARGETS),
        checks=['zero_exact','rotation_sign','history_frame_schedule','norm_preserved',
                'other_slots_unchanged','input_not_mutated','ramp','source_fixtures','checkpoint_hashes']))
    print('Pitch reference input checks passed; no physics or model inference',flush=True)


def guard(root, replay=False):
    budget=json.loads((root/'budget.json').read_text())
    if time.time()>=budget['start_unix']+(345 if replay else 590):
        raise TimeoutError('Authorized experiment deadline; no retry')


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
    disturbances=from_task_config();assert not(disturbances.action_delay or disturbances.sensor_delay or disturbances.pushes)
    policies=[load_policy(p,'cuda:0') for p in MODELS]
    assert [m['iterations'] for _,m in policies]==[8000,9000]
    assert policies[0][1]['reset_contract']==policies[1][1]['reset_contract']
    frozen=[{k:v.detach().cpu().clone() for k,v in p.state_dict().items()} for p,_ in policies]
    checks=[];prepared=[];executions=[]
    def forbidden(*args,**kwargs):raise AssertionError('Optimizer updates forbidden')
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam,'step',side_effect=forbidden))
        for height in HEIGHTS:
            guard(root,True);source=SOURCE/height/'gpu15'
            original=json.loads((source/'protocol.json').read_text())
            items=[deepcopy(next(s for s in original['samples'] if s['id']==sid)) for sid in SIDS]
            # 原地延长至15秒，所有组在10～15秒比较；停车命令仍在t=7秒下发。
            items[0]['case']['duration']=15.;items[0]['case']['segments'][0]['duration']=15.
            folders={}
            for b,branch in enumerate(BRANCHES):
                folder=root/height/branch;folder.mkdir(parents=True);folders[branch]=folder
                (folder/'fixtures').mkdir();(folder/'traces').mkdir()
                write(folder/'protocol.json',dict(protocol='fudan-pitch-reference-probe-v1',mode='diagnostic_not_formal_acceptance',
                    samples=items,validation_engine='gpu',validation_backend='mujoco_warp_gpu',
                    target_pitch_deg=TARGETS[b],measurement_window=[10.,15.],integral_enabled=False,
                    intervention='Gravity feedback reference only, not a mechanical constraint; raw dynamics and scoring use real state'))
            env=FudanEnv(18,seed=50001,randomized=True,reset_contract=policies[0][1]['reset_contract'],disturbances=disturbances)
            stack.callback(env.close)
            try:obs,_=restore_initial(env,items,source/'fixtures',folders,checks,height,branches=BRANCHES)
            finally:write(root/'initial-consistency.json',checks)
            prepared.append((height,items,env,obs))
        assert len(checks)==54 and all(c['all_arrays_identical'] for c in checks)
        write(root/'initialization-result.json',dict(passed=True,checked=54,policy_batches_before_checks=0))
        for height,items,env,obs in prepared:
            guard(root,True);env.action.record_substeps=True
            starts=obs.new_tensor([2.,7.,7.]*6)
            target=obs.new_tensor(np.repeat(np.deg2rad(TARGETS),3))
            rows=[gpu_row(env.env).cpu().numpy()];poses=[env.env.sim.data.qpos.cpu().numpy().copy()]
            terminated=[np.zeros(18,dtype=bool)];actions=[];estimates=[];angles=[];velocities=[];completed=0;error=None
            try:
                with torch.inference_mode():
                    for tick in range(1500):
                        guard(root,True);t=tick*.01
                        raw=obs.new_tensor([command(s['case'],t) for s in items]*6)
                        obs=update_evaluation_commands(env,raw,obs)
                        biased,history=policy_inputs(obs,env.history,t,starts,target)
                        requested=reference_angles(t,starts,target)
                        output=[];encoded=[]
                        for b in range(6):
                            lo,hi=b*3,(b+1)*3
                            a,e=policies[b//3][0].act_inference(biased[lo:hi],history[lo:hi])
                            output.append(a);encoded.append(e)
                        applied=torch.cat(output);estimate=torch.cat(encoded)
                        assert torch.isfinite(applied).all() and torch.isfinite(estimate).all()
                        actions.append(applied.cpu().numpy());estimates.append((estimate/2.).cpu().numpy());angles.append(requested.cpu().numpy())
                        obs,_,_,_,_,_=env.step(applied,auto_reset=False,resample=False)
                        velocities.append(env.env.sim.data.qvel[:,env.action._motor_v_adr].cpu().numpy().copy())
                        rows.extend(torch.stack(env.action.substep_rows).cpu().numpy());poses.append(env.env.sim.data.qpos.cpu().numpy().copy())
                        terminated.extend([(env.fail_ticks>100).cpu().numpy().copy()]*5);completed=tick+1
                        if tick%300==0:print(json.dumps(dict(height=height,step=completed,total=1500)),flush=True)
            except Exception as exc:error=repr(exc)
            finally:
                rows=np.asarray(rows);poses=np.asarray(poses);terminated=np.asarray(terminated)
                origins=env.env.scene.env_origins.cpu().numpy()
                for b,branch in enumerate(BRANCHES):
                    folder=root/height/branch
                    for i,sample in enumerate(items):
                        idx=b*3+i;q=poses[:,idx].copy();q[:,:3]-=origins[idx]
                        trace=arrays(rows[:,idx],q,terminated=terminated[:,idx])
                        if actions:
                            trace.update(policy_time=np.arange(completed)*.01,action=np.asarray(actions)[:completed,idx],
                                encoder_m_s=np.asarray(estimates)[:completed,idx],pitch_reference_rad=np.asarray(angles)[:completed,idx],
                                motor_qvel_rad_s=np.asarray(velocities)[:completed,idx])
                        np.savez_compressed(folder/'traces'/(sample['id']+'.npz'),**trace)
                    write(folder/'execution.json',dict(engine='mjlab',validation_backend='mujoco_warp_gpu',
                        complete_traces=3 if completed==1500 else 0,planned=3,error=error,
                        policy_batches=completed,optimizer_updates=0,integral_enabled=False))
                executions.append(dict(height=height,steps=completed,error=error));write(root/'replay-executions.json',executions)
            if error:raise RuntimeError(error)
    assert all(torch.equal(f[k],v.detach().cpu()) for f,(p,_) in zip(frozen,policies) for k,v in p.state_dict().items())
    write(root/'frozen-policies.json',dict(identical=True,optimizer_updates=0,policy_batches=27000,
        environment_physics_substeps=405000,integral_enabled=False))


def pitch_metrics(trace, target):
    pitch=trace['roll_pitch_deg'][5000:7500,1]
    err=np.abs(pitch-target)
    fraction=float((err<=.5).mean())
    return dict(target_pitch_deg=float(target),actual_pitch_mean_deg=float(pitch.mean()),
        actual_pitch_min_deg=float(pitch.min()),actual_pitch_max_deg=float(pitch.max()),
        target_error_p95_deg=float(np.quantile(err,.95)),fraction_within_half_degree=fraction,
        actual_target_attained=fraction>=.95)


def report(root):
    from .report_parking_integral import parking_metrics
    rows=[];executions=[]
    for height in HEIGHTS:
        guard(root)
        for b,branch in enumerate(BRANCHES):
            folder=root/height/branch
            command=['scripts/mjlab.sh','accept','--task','fudan_ppo_walk','--recording',str(folder),'--output',str(folder/'acceptance')]
            if b%3:
                command+=['--baseline',str(root/height/BRANCHES[b-b%3]/'acceptance/acceptance.json')]
            with (folder/'accept.log').open('w') as log:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=30)
            assert proc.returncode==0
            acceptance=json.loads((folder/'acceptance/acceptance.json').read_text())
            executions.append(dict(height=height,branch=branch,returncode=proc.returncode,technical_passed=acceptance['technical_passed']))
            for sid in SIDS:
                with np.load(folder/'traces'/(sid+'.npz')) as t:
                    metrics=parking_metrics(t,10.,15.)
                    metrics.update(pitch_metrics(t,TARGETS[b]))
                    v=t['motor_qvel_rad_s'][1000:1500,:][:,[0,1,3,4]]
                    metrics['leg_velocity_rms_rad_s']=float(np.sqrt(np.mean(v.astype(float)**2)))
                    metrics['body_contact_peak_N']=float(t['base_contact'].max())
                    metrics['full_trajectory_pitch_peak_deg']=float(np.abs(t['roll_pitch_deg'][:,1]).max())
                    metrics['terminated']=bool(t['terminated'].any())
                    metrics['finite']=all(np.isfinite(t[k]).all() for k in ('position','linear_world','roll_pitch_deg'))
                rows.append(dict(height=height,branch=branch,id=sid,**metrics))
    for row in rows:
        branch=row['branch'];baseline=branch.split('_')[0]+'_zero'
        old=next(r for r in rows if r['height']==row['height'] and r['branch']==baseline and r['id']==row['id'])
        row['leg_rms_ratio_to_baseline']=row['leg_velocity_rms_rad_s']/old['leg_velocity_rms_rad_s']
        row['signed_vx_change_from_baseline']=row['net_x_m']/5.-old['net_x_m']/5.
    write(root/'accept-cli-results.json',executions)
    write(root/'comparison.json',dict(rows=rows,measurement_window=[10,15],optimizer_updates=0,
        attainment_definition='At least95% of500Hz samples within0.5deg of the requested pitch; diagnostic only, unified5deg gate unchanged',
        interpretation='If actual pitch is not attained, this is a reference-intervention result, not a direct test of the requested physical pitch',
        final_acceptance='diagnostic_only_not_a_trained_candidate'))
    lines=[f'# {TARGETS[1]:+g}°/{TARGETS[2]:+g}°姿态参考补偿后退：冻结GPU验证','','仅改变策略当前帧和历史中的重力参考；真实物理/角速度/命令/评分不加偏置，不用外力强行固定倾角。零速时从t=2s、驾驶后从t=7s开始，以0.5秒斜坡施加参考。所有组15秒，固定10～15秒测量。积分关闭，不训练。', '',
        '**必须先检查真实pitch是否达到目标，未达到不能声称已验证对应真实倾角。**','','| 高度/模型/工况 | 参考° | 实际pitch均值° | 达到目标 | 5秒X位移cm | 均速cm/s | 腿速RMS rad/s |',
        '|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["height"]}/{r["branch"]}/{r["id"]} | {r["target_pitch_deg"]:+.0f} | {r["actual_pitch_mean_deg"]:+.2f} | {r["actual_target_attained"]} | {100*r["net_x_m"]:+.2f} | {100*r["mean_horizontal_speed_m_s"]:.2f} | {r["leg_velocity_rms_rad_s"]:.4f} |')
    lines+=['','负X为后退，均速是世界水平速度模长，不抵消往复运动。腿速来自100Hz保存的瞬时qvel，区别于上一轮100Hz位置差分；只能在本轮同口径分支比较。','','[详细数据](comparison.json) · [预算](budget.json) · [初态核对](initial-consistency.json)', '',
        f'正常速度视频，第一行8000/第二行9000，各行从左到右为0/{TARGETS[1]:+g}/{TARGETS[2]:+g}度参考：','',
        '[0.22m](videos/h022.mp4) · [0.26m](videos/h026.mp4) · [0.30m](videos/h030.mp4)']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=len(rows),target_attained=sum(r['actual_target_attained'] for r in rows if r['target_pitch_deg']),
        nonzero_target_rows=sum(bool(r['target_pitch_deg']) for r in rows))),flush=True)


def video(root):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    (root/'videos').mkdir();index=[];review=[]
    for height in HEIGHTS:
        guard(root);model=mujoco.MjModel.from_binary_path(str(root/height/BRANCHES[0]/'fixtures/model.mjb'))
        disable_model_sameframe_shortcuts(model);data=mujoco.MjData(model)
        model.vis.global_.offwidth=320;model.vis.global_.offheight=240
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        camera=mujoco.MjvCamera();camera.distance=1.65;camera.elevation=-18.;camera.azimuth=110.
        path=root/'videos'/(height+'.mp4');frames=0;chapters=[]
        with mujoco.Renderer(model,height=240,width=320) as renderer,imageio.get_writer(path,fps=25,codec='libx264',quality=7) as writer:
            for sid in SIDS:
                traces=[]
                for branch in BRANCHES:
                    with np.load(root/height/branch/'traces'/(sid+'.npz')) as f:
                        traces.append({k:f[k] for k in ('qpos','linear_world','roll_pitch_deg')})
                assert all(len(t['qpos'])==1501 for t in traces)
                start=frames
                for tick in range(0,1500,4):
                    guard(root);canvas=Image.new('RGB',(960,512),'#101820');draw=ImageDraw.Draw(canvas)
                    draw.text((6,8),f'{height} | {sid} | 8000 top / 9000 bottom | columns0 / {TARGETS[1]:+g} / {TARGETS[2]:+g}deg | real speed1x',fill='white')
                    for b,trace in enumerate(traces):
                        q=trace['qpos'][tick];data.qpos[:]=q;mujoco.mj_forward(model,data)
                        camera.lookat[:]=q[:3];camera.lookat[2]-=.025
                        renderer.update_scene(data,camera=camera,scene_option=option)
                        for x in range(int(np.floor(q[0]))-2,int(np.floor(q[0]))+3):
                            if renderer.scene.ngeom>=renderer.scene.maxgeom:break
                            geom=renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_BOX,np.array([.04,.4,.003]),np.array([float(x),0.,.003]),np.eye(3).reshape(9),np.array([.95,.7,.2,1.]))
                            renderer.scene.ngeom+=1
                        panel=Image.fromarray(renderer.render().copy());pd=ImageDraw.Draw(panel)
                        pd.rectangle((0,0,320,57),fill='#18212b');j=tick*5
                        pd.text((4,4),f'{BRANCHES[b]} | t={tick*.01:.2f}s | I=OFF',fill='white')
                        pd.text((4,22),f'real pitch={trace["roll_pitch_deg"][j,1]:+.2f}deg',fill='white')
                        pd.text((4,39),f'vx={trace["linear_world"][j,0]:+.3f}m/s X={q[0]:+.2f}m',fill='white')
                        canvas.paste(panel,((b%3)*320,32+(b//3)*240))
                    writer.append_data(np.asarray(canvas));frames+=1
                    if tick==1496:review.append(canvas.copy())
                chapters.append(dict(sample_id=sid,start_s=start/25,duration_s=(frames-start)/25,complete=True))
        guard(root);reader=imageio.get_reader(path);count=sum(1 for frame in reader);meta=reader.get_meta_data();reader.close()
        assert count==frames==1125 and meta['fps']==25
        item=dict(height=height,path=str(path),frames=frames,fps=25,duration_s=frames/25,chapters=chapters,sha256=sha(path),all_frames_decoded=True)
        index.append(item);write(root/'video-index.json',index);print(json.dumps(item),flush=True)
    sheet=Image.new('RGB',(960*3,512*3),'#101820')
    for i,im in enumerate(review):sheet.paste(im,((i//3)*960,(i%3)*512))
    sheet.save(root/'video-review.png')


def pipeline(root):
    checks=json.loads((root/'checks.json').read_text())
    assert checks['passed'] and checks['reference_degrees']==list(TARGETS)
    assert not(root/'budget.json').exists(),'Never reuse a started window'
    sources=[*Path('mjlab_training/src/ct_mjlab').rglob('*.py'),*Path('scripts/agent/fudan_ppo_walk').glob('*.py'),
        *Path('scripts/agent/motion').glob('*.py'),Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')]
    manifest={str(p):sha(p) for p in sources}
    inputs={str(p):sha(p) for p in SOURCE.glob('h*/gpu15/fixtures/*') if p.is_file()}
    for p in sources:
        target=root/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(root/'source-sha256.json',manifest);write(root/'input-sha256.json',inputs)
    started=time.time();mono=time.monotonic()
    budget=dict(start_unix=started,deadline_unix=started+600,maximum_wall_s=600,replay_maximum_wall_s=360,
        start_local=datetime.fromtimestamp(started,ZoneInfo('Asia/Shanghai')).isoformat(),
        deadline_local=datetime.fromtimestamp(started+600,ZoneInfo('Asia/Shanghai')).isoformat(),
        authorization=f'用户选中建议要求“测”{TARGETS[1]:+g}/{TARGETS[2]:+g}度；比较8000和9000，沿用上一轮已确认的单次600秒上限；不训练、不追加角度、不重试',
        reference_degrees=list(TARGETS),
        models=[str(p) for p in MODELS],model_sha256=HASHES,expected_trajectories=54,
        protocol='三高度×原地/后退停车/前进停车×两模型×三参考；原地延长15秒，固定10～15秒测量，真实pitch达标独立核对')
    write(root/'budget.json',budget);print(json.dumps(budget,ensure_ascii=False),flush=True)
    env=os.environ.copy();env.update(PYTHONPATH='mjlab_training/src',MUJOCO_GL='egl');steps=[]
    for stage,limit in (('replay',360),('report',600),('video',600)):
        remaining=limit-(time.monotonic()-mono)
        if remaining<=0:break
        with (root/(stage+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-m','scripts.agent.fudan_ppo_walk.probe_pitch_reference',stage,'--output',str(root),
                '--reference-degrees',str(int(TARGETS[1])),str(int(TARGETS[2]))],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            timeout=False
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        steps.append(dict(stage=stage,returncode=code,hard_timeout=timeout,finished_elapsed_s=time.monotonic()-mono))
        write(root/'pipeline-progress.json',steps);print(json.dumps(steps[-1]),flush=True)
        if code:break
    result=dict(steps=steps,wall_s=time.monotonic()-mono,finished_unix=time.time(),optimizer_updates=0,no_retry=True,
        models_unchanged=all(sha(p)==h for p,h in zip(MODELS,HASHES)),source_unchanged=all(sha(p)==h for p,h in manifest.items()),
        inputs_unchanged=all(sha(p)==h for p,h in inputs.items()),within_budget=time.monotonic()-mono<=600,
        final_acceptance='diagnostic_only_not_a_trained_candidate')
    write(root/'pipeline-result.json',result);print(json.dumps(result),flush=True)


def main():
    global BRANCHES,TARGETS
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=('check','pipeline','replay','report','video'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-degrees',type=int,nargs=2,default=(5,-5))
    args=parser.parse_args()
    TARGETS=(0.,float(args.reference_degrees[0]),float(args.reference_degrees[1]))*2
    def label(value):return 'zero' if value==0 else ('plus' if value>0 else 'minus')+f'{abs(value):g}'
    BRANCHES=tuple(f'{model}_{label(value)}' for model in (8000,9000) for value in TARGETS[:3])
    globals()[args.mode](args.output)


if __name__=='__main__':main()
