"""Bounded paired GPU parking comparison: selected8000+2deg, zero versus4+4ms."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_sim2sim_config
from ct_mjlab.fudan_ppo_walk.selected_policy import selection_record,reference_starts,policy_inputs
from .diagnose_parking_integral import restore_initial

HEIGHTS=('h022','h026','h030')
BRANCHES=('baseline','delayed')
SIDS=('diagnostic_0','diagnostic_13','diagnostic_14')
SOURCE=Path('artifacts/fudan_ppo_walk/fudan-gpu-replay-8000-20261102')
ALLOWED=('action_delay_steps','sensor_delay_steps')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def items_for(height):
    original=json.loads((SOURCE/height/'gpu15/protocol.json').read_text())
    items=[deepcopy(next(s for s in original['samples'] if s['id']==sid)) for sid in SIDS]
    items[0]['case']['duration']=15.;items[0]['case']['segments'][0]['duration']=15.
    return items


def conditions():
    delayed=from_sim2sim_config()
    assert delayed.action_delay and delayed.sensor_delay and delayed.action_delay_ms==delayed.sensor_delay_ms==(4.,4.) and not delayed.pushes
    return dict(baseline=replace(delayed,action_delay=False,sensor_delay=False),delayed=delayed)


def verify_pair(folders,items):
    results=[]
    for sample in items:
        name=sample['id']+'_initial.npz'
        with np.load(folders['baseline']/'fixtures'/name) as a,np.load(folders['delayed']/'fixtures'/name) as b:
            assert set(a.files)==set(b.files)
            unchanged=[];changed=[]
            for key in a.files:
                assert a[key].shape==b[key].shape and a[key].dtype==b[key].dtype,key
                if key in ALLOWED:
                    assert int(a[key])==0 and int(b[key])==2,key
                    changed.append(key)
                else:
                    assert np.array_equal(a[key],b[key]),key
                    unchanged.append(key)
            results.append(dict(sample_id=sample['id'],all_physical_arrays_identical=True,unchanged_keys=unchanged,
                declared_changed_keys=changed,baseline_steps=[0,0],delayed_steps=[2,2]))
    return results


def check(root):
    from .check_disturbances import control_fixture
    from ct_mjlab.fudan_ppo_walk.measurement import FIELDS
    from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT
    root.mkdir(parents=True,exist_ok=False);cfgs=conditions();selection=selection_record();checks=[];initial=[]
    def forbidden(*a,**kw):raise AssertionError('Real physics, NN and optimization forbidden in preparation')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as directory:
        for target in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','mjlab.envs.ManagerBasedRlEnv.step','torch.nn.Linear.forward','torch.optim.Adam.step','torch.cuda._lazy_init'):
            stack.enter_context(patch(target,side_effect=forbidden))
        for height in HEIGHTS:
            items=items_for(height);folders={};inputs=[]
            for branch in BRANCHES:
                folder=Path(directory)/height/branch;(folder/'fixtures').mkdir(parents=True);folders[branch]=folder
                env=control_fixture(cfgs[branch]);env.reset_contract=RESET_V2_CONTRACT
                with np.load(SOURCE/height/'gpu15/fixtures'/('diagnostic_0_initial.npz')) as archive:
                    for key in FIELDS:setattr(env.env.sim.model,key,torch.full((3,*archive[key].shape),99.,dtype=torch.float32))
                env.action._nominal_pos.add_(.02);env.env.sim.data.qpos[:,7:]+=.01;env.history.fill_(99.)
                obs,_=restore_initial(env,items,SOURCE/height/'gpu15/fixtures',{branch:folder},initial,height,
                    branches=(branch,),delay_steps=(0,0) if branch=='baseline' else (2,2))
                assert env.mock_physics_calls==0
                assert env.action.disturbances.action_steps.eq(0 if branch=='baseline' else 2).all()
                assert env.action.disturbances.sensor_steps.eq(0 if branch=='baseline' else 2).all()
                inputs.append((obs.clone(),env.history.clone()))
            pair=verify_pair(folders,items);assert len(pair)==3
            assert all(torch.equal(a,b) for a,b in zip(inputs[0],inputs[1]))
            checks.append(height+': physical inputs exact, declared delay steps only, initial obs/history exact')
        assert len(initial)==18 and all(r['all_arrays_identical'] for r in initial)
    write(root/'checks.json',dict(passed=True,checks=checks,initial_checks=18,selection=selection,
        conditions={k:v.record() for k,v in cfgs.items()},physics_steps=0,model_forwards=0,optimizer_updates=0))
    print('Paired restoration of18 archived cases passed; physical inputs exact; only delay steps differ',flush=True)


def guard(root,replay=False):
    budget=json.loads((root/'budget.json').read_text())
    if time.time()>=budget['start_unix']+(345 if replay else 590):raise TimeoutError('Authorized experiment deadline')


def replay(root):
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    from ct_mjlab.fudan_ppo_walk.measurement import gpu_row,arrays,update_evaluation_commands
    from .evaluate import command
    torch.set_num_threads(1);selection=selection_record();cfgs=conditions()
    policy,metadata=load_policy(Path(selection['checkpoint']),'cuda:0');assert metadata['iterations']==8000
    frozen={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    initial=[];pairs=[];prepared=[];executions=[]
    def forbidden(*a,**kw):raise AssertionError('Optimizer updates forbidden')
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam,'step',side_effect=forbidden))
        for height in HEIGHTS:
            guard(root,True);items=items_for(height);folders={};inputs=[]
            for branch in BRANCHES:
                folder=root/height/branch;folder.mkdir(parents=True);folders[branch]=folder
                (folder/'fixtures').mkdir();(folder/'traces').mkdir()
                write(folder/'protocol.json',dict(protocol='fudan-parking-delay-pair-v1',mode='diagnostic_not_formal_acceptance',samples=items,
                    validation_engine='gpu',validation_backend='mujoco_warp_gpu',selected_policy=selection,
                    disturbances=cfgs[branch].record(),training_disturbances=metadata['disturbances'],measurement_window=[10,15],target_pitch_deg=2.))
                env=FudanEnv(3,seed=50001,randomized=True,reset_contract=metadata['reset_contract'],disturbances=cfgs[branch]);stack.callback(env.close)
                try:obs,_=restore_initial(env,items,SOURCE/height/'gpu15/fixtures',{branch:folder},initial,height,
                    branches=(branch,),delay_steps=(0,0) if branch=='baseline' else (2,2))
                finally:write(root/'initial-consistency.json',initial)
                inputs.append((obs.clone(),env.history.clone()));prepared.append((height,branch,items,env,obs))
            assert all(torch.equal(a,b) for a,b in zip(inputs[0],inputs[1]))
            pairs.extend(dict(height=height,**p) for p in verify_pair(folders,items))
        assert len(initial)==18 and all(p['all_arrays_identical'] for p in initial) and len(pairs)==9
        write(root/'physical-pairing.json',pairs);write(root/'initialization-result.json',dict(passed=True,checked=18,physical_pairs=9,policy_batches_before_checks=0))
        for height,branch,items,env,obs in prepared:
            guard(root,True);env.action.record_substeps=True;starts=reference_starts(items,obs);target=obs.new_full((3,),np.deg2rad(2.))
            rows=[gpu_row(env.env).cpu().numpy()];poses=[env.env.sim.data.qpos.cpu().numpy().copy()];terminated=[np.zeros(3,dtype=bool)]
            velocities=[];actions=[];estimates=[];completed=0;error=None
            try:
                with torch.inference_mode():
                    for tick in range(1500):
                        guard(root,True);t=tick*.01
                        obs=update_evaluation_commands(env,[command(s['case'],t) for s in items],obs)
                        actor_obs,history=policy_inputs(obs,env.history,t,starts,target)
                        action,estimate=policy.act_inference(actor_obs,history)
                        assert torch.isfinite(action).all() and torch.isfinite(estimate).all()
                        actions.append(action.cpu().numpy());estimates.append((estimate/2.).cpu().numpy())
                        obs,_,_,_,_,_=env.step(action,auto_reset=False,resample=False)
                        velocities.append(env.env.sim.data.qvel[:,env.action._motor_v_adr].cpu().numpy().copy())
                        rows.extend(torch.stack(env.action.substep_rows).cpu().numpy());poses.append(env.env.sim.data.qpos.cpu().numpy().copy())
                        terminated.extend([(env.fail_ticks>100).cpu().numpy().copy()]*5);completed=tick+1
                        if tick%300==0:print(json.dumps(dict(height=height,branch=branch,step=completed,total=1500)),flush=True)
            except Exception as exc:error=repr(exc)
            finally:
                rows=np.asarray(rows);poses=np.asarray(poses);terminated=np.asarray(terminated);origins=env.env.scene.env_origins.cpu().numpy()
                folder=root/height/branch
                for i,sample in enumerate(items):
                    q=poses[:,i].copy();q[:,:3]-=origins[i]
                    trace=arrays(rows[:,i],q,terminated=terminated[:,i])
                    if completed:trace.update(policy_time=np.arange(completed)*.01,motor_qvel_rad_s=np.asarray(velocities)[:completed,i],
                        action=np.asarray(actions)[:completed,i],encoder_m_s=np.asarray(estimates)[:completed,i])
                    np.savez_compressed(folder/'traces'/(sample['id']+'.npz'),**trace)
                write(folder/'execution.json',dict(engine='mjlab',validation_backend='mujoco_warp_gpu',complete_traces=3 if completed==1500 else 0,
                    planned=3,policy_batches=completed,optimizer_updates=0,error=error))
                executions.append(dict(height=height,branch=branch,steps=completed,error=error));write(root/'replay-executions.json',executions)
            if error:raise RuntimeError(error)
    assert all(torch.equal(v,policy.state_dict()[k].detach().cpu()) for k,v in frozen.items())
    write(root/'frozen-policy.json',dict(identical=True,optimizer_updates=0,actor_batches=9000,encoder_batches=9000,environment_physics_substeps=135000))


def report(root):
    from .report_parking_integral import parking_metrics
    rows=[];acceptances={};executions=[]
    for height in HEIGHTS:
        for branch in BRANCHES:
            guard(root);folder=root/height/branch
            with (folder/'accept.log').open('w') as log:
                proc=subprocess.run(['scripts/mjlab.sh','accept','--task','fudan_ppo_walk','--recording',str(folder),'--output',str(folder/'acceptance')],stdout=log,stderr=subprocess.STDOUT,timeout=30)
            assert proc.returncode==0
            acceptance=json.loads((folder/'acceptance/acceptance.json').read_text());acceptances[(height,branch)]=acceptance
            executions.append(dict(height=height,branch=branch,returncode=proc.returncode,technical_passed=acceptance['technical_passed']))
            for sid in SIDS:
                with np.load(folder/'traces'/(sid+'.npz')) as trace:
                    metrics=parking_metrics(trace,10.,15.)
                    metrics.update(leg_velocity_rms_rad_s=float(np.sqrt(np.mean(trace['motor_qvel_rad_s'][1000:1500,:][:,[0,1,3,4]].astype(float)**2))),
                        pitch_peak_deg=float(np.abs(trace['roll_pitch_deg'][5000:7500,1]).max()),
                        pitch_mean_deg=float(trace['roll_pitch_deg'][5000:7500,1].mean()),
                        base_contact_peak_N=float(trace['base_contact'].max()),terminated=bool(trace['terminated'].any()))
                if branch=='baseline':
                    with np.load(Path('artifacts/fudan_ppo_walk/fudan-pitch-reference-small-20261103')/height/'8000_plus2/traces'/(sid+'.npz')) as old:
                        previous=parking_metrics(old,10.,15.)
                    metrics['previous_baseline_vector_difference_m']=float(np.linalg.norm(np.array([metrics['net_x_m'],metrics['net_y_m']])-np.array([previous['net_x_m'],previous['net_y_m']])))
                rows.append(dict(height=height,branch=branch,id=sid,**metrics))
    paired=[];regressed=[]
    for row in [r for r in rows if r['branch']=='delayed']:
        baseline=next(r for r in rows if r['height']==row['height'] and r['id']==row['id'] and r['branch']=='baseline')
        paired.append(dict(height=row['height'],id=row['id'],offset_change_m=row['maximum_horizontal_offset_m']-baseline['maximum_horizontal_offset_m'],
            leg_rms_ratio=row['leg_velocity_rms_rad_s']/baseline['leg_velocity_rms_rad_s'],strict_parking_lost=baseline['passed'] and not row['passed']))
    for height in HEIGHTS:
        old={r['sample_id']:r['passed'] for r in acceptances[(height,'baseline')]['samples']}
        regressed.extend(height+'/'+r['sample_id'] for r in acceptances[(height,'delayed')]['samples'] if old[r['sample_id']] and not r['passed'])
    summary=dict(parking_passed={b:sum(r['passed'] for r in rows if r['branch']==b) for b in BRANCHES},
        unified_passed={b:sum(acceptances[(h,b)]['technical_passed'] for h in HEIGHTS) for b in BRANCHES},
        unified_regressed=regressed,baseline_reproduced=all(r['previous_baseline_vector_difference_m']<=.02 for r in rows if r['branch']=='baseline'),
        rows=rows,pairs=paired,paired_on='All decoded initial arrays exact except declared action/sensor steps0to2; standard cohort comparison not bypassed',
        optimizer_updates=0,final_acceptance='pending_user_video_review')
    write(root/'comparison.json',summary);write(root/'accept-cli-results.json',executions)
    lines=['# 8000＋停车2°：零延迟对照动作4ms＋传感4ms','','所有组固定15秒，10～15秒测量，积分关闭。只改变两项延迟；物理初态逐值相同，零延迟基线重新运行。', '',
        '| 高度/工况 | 零延迟：偏移cm / 均速cm/s / 腿RMS | 4+4ms |','|---|---:|---:|']
    for height in HEIGHTS:
        for sid in SIDS:
            values=[next(r for r in rows if r['height']==height and r['id']==sid and r['branch']==b) for b in BRANCHES]
            lines.append('| '+height+'/'+sid+' | '+' | '.join(f"{r['maximum_horizontal_offset_m']*100:.2f} / {r['mean_horizontal_speed_m_s']*100:.3f} / {r['leg_velocity_rms_rad_s']:.4f}" for r in values)+' |')
    lines+=['',f"严格停车：{summary['parking_passed']}；统一技术：{summary['unified_passed']}；统一原成功丢失：{regressed}。",'',
        '严格停车为5秒最大偏移≤5cm、世界水平速度模长均值≤0.01m/s、峰值≤0.05m/s。腿RMS来自100Hz保存的瞬时关节速度，不能由姿态/速度门槛代替平滑性验收。', '',
        '[详细比较](comparison.json) · [物理初态配对](physical-pairing.json) · [预算](budget.json)', '',
        '正常速度完整视频，左零延迟、右4+4ms：[0.22m](videos/h022.mp4) · [0.26m](videos/h026.mp4) · [0.30m](videos/h030.mp4)。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('parking_passed','unified_passed','unified_regressed','baseline_reproduced')},ensure_ascii=False),flush=True)


def video(root):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    (root/'videos').mkdir();index=[];review=[]
    for height in HEIGHTS:
        guard(root);model=mujoco.MjModel.from_binary_path(str(root/height/'baseline/fixtures/model.mjb'))
        disable_model_sameframe_shortcuts(model);data=mujoco.MjData(model)
        model.vis.global_.offwidth=384;model.vis.global_.offheight=288
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        camera=mujoco.MjvCamera();camera.distance=1.65;camera.elevation=-18.;camera.azimuth=110.
        path=root/'videos'/(height+'.mp4');frames=0;chapters=[]
        with mujoco.Renderer(model,height=288,width=384) as renderer,imageio.get_writer(path,fps=50,codec='libx264',quality=7) as writer:
            for sid in SIDS:
                traces=[]
                for b in BRANCHES:
                    with np.load(root/height/b/'traces'/(sid+'.npz')) as f:traces.append({k:f[k] for k in ('qpos','linear_world','roll_pitch_deg')})
                assert all(len(t['qpos'])==1501 for t in traces)
                start=frames
                for tick in range(0,1500,2):
                    guard(root);canvas=Image.new('RGB',(768,320),'#101820');draw=ImageDraw.Draw(canvas)
                    draw.text((6,8),f'8000 +2deg | {height} | {sid} | LEFT 0ms / RIGHT action4ms+sensor4ms | normal1x',fill='white')
                    for b,trace in enumerate(traces):
                        q=trace['qpos'][tick];data.qpos[:]=q;mujoco.mj_forward(model,data)
                        camera.lookat[:]=q[:3];camera.lookat[2]-=.025;renderer.update_scene(data,camera=camera,scene_option=option)
                        for x in range(int(np.floor(q[0]))-2,int(np.floor(q[0]))+3):
                            if renderer.scene.ngeom>=renderer.scene.maxgeom:break
                            geom=renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_BOX,np.array([.04,.4,.003]),np.array([float(x),0.,.003]),np.eye(3).reshape(9),np.array([.95,.7,.2,1.]))
                            renderer.scene.ngeom+=1
                        panel=Image.fromarray(renderer.render().copy());pd=ImageDraw.Draw(panel);pd.rectangle((0,0,384,57),fill='#18212b');j=tick*5
                        pd.text((5,4),f'{BRANCHES[b]} | t={tick*.01:.2f}s | I=OFF',fill='white')
                        pd.text((5,22),f'speed={np.linalg.norm(trace["linear_world"][j,:2]):.3f}m/s pitch={trace["roll_pitch_deg"][j,1]:+.2f}deg',fill='white')
                        pd.text((5,39),f'world X={q[0]:+.3f}m',fill='white');canvas.paste(panel,(384*b,32))
                    writer.append_data(np.asarray(canvas));frames+=1
                    if tick==1498:review.append(canvas.copy())
                chapters.append(dict(sample_id=sid,start_s=start/50,duration_s=(frames-start)/50,complete=True))
        guard(root);reader=imageio.get_reader(path);count=sum(1 for frame in reader);meta=reader.get_meta_data();reader.close()
        assert count==frames==2250 and meta['fps']==50
        index.append(dict(height=height,path=str(path),frames=frames,fps=50,duration_s=frames/50,chapters=chapters,sha256=sha(path),all_frames_decoded=True))
        write(root/'video-index.json',index);print(json.dumps(index[-1]),flush=True)
    sheet=Image.new('RGB',(768*3,320*3),'#101820')
    for i,im in enumerate(review):sheet.paste(im,((i//3)*768,(i%3)*320))
    sheet.save(root/'video-review.png')


def pipeline(root):
    checks=json.loads((root/'checks.json').read_text());assert checks['passed'] and checks['selection']==selection_record()
    assert not(root/'budget.json').exists(),'Never reuse a started window'
    sources=[*Path('mjlab_training/src/ct_mjlab').rglob('*.py'),*Path('scripts/agent/fudan_ppo_walk').glob('*.py'),*Path('scripts/agent/motion').glob('*.py'),Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')]
    manifest={str(p):sha(p) for p in sources};inputs={str(p):sha(p) for p in SOURCE.glob('h*/gpu15/fixtures/*') if p.is_file()}
    for p in sources:
        target=root/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(root/'source-sha256.json',manifest);write(root/'input-sha256.json',inputs)
    started=time.time();mono=time.monotonic();selection=selection_record()
    budget=dict(start_unix=started,deadline_unix=started+600,maximum_wall_s=600,replay_maximum_wall_s=360,
        start_local=datetime.fromtimestamp(started,ZoneInfo('Asia/Shanghai')).isoformat(),deadline_local=datetime.fromtimestamp(started+600,ZoneInfo('Asia/Shanghai')).isoformat(),
        authorization='用户选中重跑九停车要求“做”；沿用单次600秒上限；冻结8000+2，零延迟与4+4ms配对，不训练/调参/重试',
        selection=selection,conditions={k:v.record() for k,v in conditions().items()},expected_trajectories=18,measurement_window=[10,15])
    write(root/'budget.json',budget);print(json.dumps(budget,ensure_ascii=False),flush=True)
    env=os.environ.copy();env.update(PYTHONPATH='mjlab_training/src',MUJOCO_GL='egl');steps=[]
    for stage,limit in (('replay',360),('report',600),('video',600)):
        remaining=limit-(time.monotonic()-mono)
        if remaining<=0:break
        with (root/(stage+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-m','scripts.agent.fudan_ppo_walk.compare_sim2sim_delays',stage,'--output',str(root)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            timeout=False
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        steps.append(dict(stage=stage,returncode=code,hard_timeout=timeout,finished_elapsed_s=time.monotonic()-mono));write(root/'pipeline-progress.json',steps);print(json.dumps(steps[-1]),flush=True)
        if code:break
    result=dict(steps=steps,wall_s=time.monotonic()-mono,finished_unix=time.time(),optimizer_updates=0,no_retry=True,
        model_unchanged=sha(selection['checkpoint'])==selection['checkpoint_sha256'],source_unchanged=all(sha(p)==h for p,h in manifest.items()),
        inputs_unchanged=all(sha(p)==h for p,h in inputs.items()),within_budget=time.monotonic()-mono<=600,final_acceptance='pending_user_video_review')
    write(root/'pipeline-result.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('check','pipeline','replay','report','video'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.mode](a.output)


if __name__=='__main__':main()
