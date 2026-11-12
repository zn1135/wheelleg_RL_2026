"""Paired GPU parking diagnosis; reuses the evaluator without changing training defaults."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_sim2sim_config
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.fudan_ppo_walk.protocol import samples
from scripts.agent.fudan_ppo_walk import evaluate


def parking_items(*_):
    # 保留原命令时序，三档高度各测静止和正反向行驶后停车。
    selected = [s for s in samples()[::21]
                if s['case']['category'] in ('zero', 'stop_forward', 'stop_backward')]
    return [dict(s, id=f'parking_{i}', seed=30001+i) for i, s in enumerate(selected)]


def render_closeups(output):
    """Side-view replay of saved GPU states; no integration or policy inference."""
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    run=json.loads((output/'run.json').read_text())
    label=Path(run['checkpoint']).stem+f' P{run["leg_p"]:g}/D{run["leg_d"]:g}'
    (output/'video').mkdir(exist_ok=True)
    selections=[('no_external','parking_5'),('configured_external','parking_7')]
    for group,sample_id in selections:
        folder=output/group
        model=mujoco.MjModel.from_binary_path(str(folder/'fixtures'/'model.mjb'))
        disable_model_sameframe_shortcuts(model)
        model.vis.global_.offwidth=960;model.vis.global_.offheight=640
        data=mujoco.MjData(model)
        with np.load(folder/'traces'/(sample_id+'.npz')) as saved:
            trace={k:saved[k].copy() for k in saved.files}
        sample=next(s for s in parking_items() if s['id']==sample_id)
        camera=mujoco.MjvCamera();camera.distance=1.6;camera.elevation=-12.;camera.azimuth=90.
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        target=output/'video'/f'{group}_{sample_id}_side.mp4'
        with mujoco.Renderer(model,height=640,width=960) as renderer, imageio.get_writer(target,fps=50,codec='libx264',quality=8) as writer:
            for i in range(0,len(trace['qpos'])-1,2):
                data.qpos[:]=trace['qpos'][i]
                mujoco.mj_forward(model,data)
                # 跟随相机用于看清倾角；同时显示世界位移和速度，不能据画面静止认定停稳。
                camera.lookat[:]=data.qpos[:3]
                renderer.update_scene(data,camera=camera,scene_option=option)
                canvas=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(canvas)
                tick=i*5;t=i*.01;cmd=evaluate.command(sample['case'],t)
                pitch=trace['roll_pitch_deg'][tick,1]
                speed=np.linalg.norm(trace['linear_world'][tick,:2])
                draw.rectangle((0,0,960,82),fill='#18212b')
                draw.text((12,8),f'{label} | {group} | {sample_id} | t={t:.2f}s | normal speed / following camera',fill='white')
                draw.text((12,31),f'cmd vx/yaw/h={cmd} | actual vx={trace["linear_body"][tick,0]:+.3f}m/s | horizontal speed={speed:.3f}m/s',fill='white')
                draw.text((12,54),f'pitch={pitch:+.2f}deg (+forward) | actual h={trace["position"][tick,2]:.3f}m | world x={trace["position"][tick,0]:+.3f}m',fill='white')
                writer.append_data(np.asarray(canvas))
        target.with_suffix('.json').write_text(json.dumps(dict(fps=50,duration_s=15.,camera='following side view',
            trajectory=str(folder/'traces'/(sample_id+'.npz')),video_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n')


def report(output):
    rows = []
    for group in ('no_external', 'configured_external'):
        folder = output/group
        protocol = json.loads((folder/'protocol.json').read_text())
        for sample in protocol['samples']:
            path = folder/'traces'/(sample['id']+'.npz')
            if not path.exists():
                rows.append(dict(group=group, case=sample['case']['id'], complete=False))
                continue
            with np.load(path) as data:
                end = sample['case']['duration']
                # 固定末5秒只作姿态诊断；正式停车窗口仍由统一accept入口评分。
                mask = (data['time'] >= end-5) & (data['time'] < end)
                complete = bool(data['time'][-1] >= end-.002 and mask.sum() == 2500)
                pitch = data['roll_pitch_deg'][mask, 1]
                speed = np.linalg.norm(data['linear_world'][mask, :2], axis=1)
                rows.append(dict(group=group, case=sample['case']['id'], sample_id=sample['id'],
                    complete=complete, window_s=[end-5, end],
                    pitch_mean_deg=float(pitch.mean()) if len(pitch) else None,
                    pitch_min_deg=float(pitch.min()) if len(pitch) else None,
                    pitch_max_deg=float(pitch.max()) if len(pitch) else None,
                    pitch_abs_peak_deg=float(np.abs(pitch).max()) if len(pitch) else None,
                    horizontal_speed_peak_m_s=float(speed.max()) if len(speed) else None,
                    horizontal_speed_mean_m_s=float(speed.mean()) if len(speed) else None,
                    final_second_pitch_mean_deg=float(data['roll_pitch_deg'][(data['time'] >= end-1) & (data['time'] < end), 1].mean()),
                    numerical_finite=bool(all(np.isfinite(data[k]).all() for k in ('position','linear_world','roll_pitch_deg')))))
    paired = []
    for sample in parking_items():
        name = sample['id']+'_initial.npz'
        with np.load(output/'no_external'/'fixtures'/name) as a, np.load(output/'configured_external'/'fixtures'/name) as b:
            # 两组唯一预期不同的是外载及推扰状态；其余初态逐数组核对。
            keys = sorted(set(a.files)-{'episode_wrench_body','push_seed','push_force_scale'})
            paired.append(dict(sample_id=sample['id'], equal=all(np.array_equal(a[k], b[k]) for k in keys), fields=keys,
                no_external_wrench=a['episode_wrench_body'].tolist(), configured_wrench=b['episode_wrench_body'].tolist()))
    result = dict(scope='diagnostic only; fixed last-five-second window, not full acceptance', rows=rows, paired_initial_states=paired)
    (output/'posture-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    colors = {'zero':'#777777', 'stop_forward':'#0072b2', 'stop_backward':'#d55e00'}
    for row, height in enumerate((.22,.26,.30)):
        for group, style in (('no_external','-'), ('configured_external','--')):
            for sample in parking_items():
                if sample['case']['segments'][0]['command'][2] != height:
                    continue
                with np.load(output/group/'traces'/(sample['id']+'.npz')) as data:
                    name=sample['case']['category']; label=name+' / '+group
                    axes[row,0].plot(data['time'],data['roll_pitch_deg'][:,1],style,color=colors[name],label=label,lw=1)
                    axes[row,1].plot(data['time'],np.linalg.norm(data['linear_world'][:,:2],axis=1),style,color=colors[name],lw=1)
        for col in (0,1):
            axes[row,col].axvline(7,color='black',lw=.7)
            axes[row,col].grid(alpha=.2)
            axes[row,col].set_title(f'height {height:.2f}m / '+('pitch (deg)' if col==0 else 'horizontal speed (m/s)'))
        axes[row,0].axhline(0,color='black',lw=.7)
        axes[row,0].axhline(5,color='red',lw=.7); axes[row,0].axhline(-5,color='red',lw=.7)
        axes[row,1].axhline(.05,color='red',lw=.7)
    axes[0,0].legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel('time (s); stop command starts at 7s')
    fig.tight_layout(); fig.savefig(output/'parking-posture.png',dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--leg-p',type=float,choices=(30.,60.),default=30.,help='Model-matched stiffness, confined to this diagnosis process')
    p.add_argument('--leg-d',type=float,choices=(2.,10.),default=2.,help='Model-matched damping, confined to this diagnosis process')
    p.add_argument('--wheel-p',type=float,choices=(.1,.045),default=.1)
    p.add_argument('--wheel-motor-gain',type=float,default=1.,help='Output multiplier after the PD torque limit')
    p.add_argument('--wheel-model',choices=('legacy_curve','mechanical'),default='legacy_curve',help='Historical diagnosis keeps its old wheel plant unless explicitly selected')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--render-only',action='store_true')
    args=p.parse_args()
    if args.render_only:
        render_closeups(args.output); return
    if args.report_only:
        report(args.output); return
    if (FudanWalkCfg.control.leg_stiffness,FudanWalkCfg.control.leg_damping)!=(30.,2.):
        raise ValueError('This diagnosis requires actual shared P30/D2 configuration')
    args.output.mkdir(parents=True,exist_ok=False)
    configured=from_sim2sim_config()
    clean=replace(configured,pushes=False,external_force_range=(0.,0.),external_torque_range=(0.,0.))
    start=time.time();deadline=start+args.seconds
    record=dict(checkpoint=str(args.checkpoint.resolve()),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        started_unix=start,deadline_unix=deadline,max_wall_seconds=args.seconds,leg_p=args.leg_p,leg_d=args.leg_d,
        wheel_p=args.wheel_p,wheel_motor_gain=args.wheel_motor_gain,
        wheel_model=args.wheel_model,
        samples=parking_items(),no_external=clean.record(),configured_external=configured.record(),
        policy_updates=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    torch.set_num_threads(1)
    original_configure=evaluate.configure_samples
    def configure_and_record(env,items,folder):
        result=original_configure(env,items,folder)
        control=dict(leg_p=env.action.cfg.leg_stiffness,leg_d=env.action.cfg.leg_damping,
            wheel_p=env.action.cfg.wheel_stiffness,wheel_d=env.action.cfg.wheel_damping,
            wheel_motor_gain=env.action.wheel_motor.gain.cpu().tolist(),
            leg_armature=env.env.sim.model.dof_armature[:,env.action._motor_v_adr[[0,1,3,4]]].cpu().tolist(),
            kp_scale=env.action.kp_scale.cpu().tolist(),kd_scale=env.action.kd_scale.cpu().tolist())
        if (control['leg_p'],control['leg_d'])!=(args.leg_p,args.leg_d):
            raise ValueError('Actual GPU action PD differs from requested model conditions')
        assert control['wheel_p']==args.wheel_p and env.action.wheel_motor.gain.eq(args.wheel_motor_gain).all()
        (Path(folder)/'actual-control.json').write_text(json.dumps(control,indent=2)+'\n')
        return result
    for name,disturbances in (('no_external',clean),('configured_external',configured)):
        if time.time() >= deadline:
            raise TimeoutError('Shared GPU diagnosis budget exhausted')
        request=SimpleNamespace(checkpoint=args.checkpoint,output=args.output/name,deadline=deadline,
            diagnostic=True,diagnostic_height=None,batch_size=9,selection=None,leg_load_profile='legacy',response_parameters=None)
        # 仅在本诊断进程内替换样本选择和扰动注入；真实GPU采集使用既有入口。
        with patch.object(evaluate,'evaluation_items',parking_items), patch.object(evaluate,'from_sim2sim_config',return_value=disturbances), patch.object(FudanWalkCfg.control,'leg_stiffness',args.leg_p), patch.object(FudanWalkCfg.control,'leg_damping',args.leg_d), patch.object(FudanWalkCfg.control,'wheel_stiffness',args.wheel_p), patch.object(FudanWalkCfg.sim,'wheel_motor_gain',args.wheel_motor_gain),patch.object(FudanWalkCfg.sim,'wheel_model',args.wheel_model), patch.object(evaluate,'configure_samples',configure_and_record):
            evaluate.gpu(request)
    record['finished_unix']=time.time();record['wall_s']=record['finished_unix']-start
    (args.output/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    report(args.output)


if __name__=='__main__':
    main()
