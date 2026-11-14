"""Paired GPU replay of source3000/P60D2 and continuation3500/P20D1."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_sim2sim_config
from scripts.agent.fudan_ppo_walk import evaluate
from scripts.agent.fudan_ppo_walk.diagnose_parking_posture import parking_items
from scripts.agent.real2sim.replay_usb_pd_targets import sha,write

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts/fudan_ppo_walk/models/20261113_model3000_p60d2_armature2_wheelP01_mechanical.pt'
NEW=ROOT/'logs/mjlab/fudan_ppo_walk/20261113_p20d1_wheelP01_mechanical_resume3000_to3500_01/model_final.pt'


def items(*_):
    result=deepcopy(parking_items())
    for row in result:
        if row['case']['category']=='zero':row['case']['duration']=30.;row['case']['segments'][0]['duration']=30.
    return result


def run(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    conditions=replace(from_sim2sim_config(),pushes=False,external_force_range=(0.,0.),external_torque_range=(0.,0.))
    plan=dict(models=[dict(name='model3000_P60D2',checkpoint=str(OLD),sha256=sha(OLD),p=60.,d=2.),
                      dict(name='model3500_P20D1',checkpoint=str(NEW),sha256=sha(NEW),p=20.,d=1.)],
        samples=items(),disturbances=conditions.record(),physics_dt_s=.002,policy_dt_s=.01,training_updates=0,
        scope='Nine nominal zero/forward-stop/backward-stop cases; same initial fixtures, no external force; not full987 or real-robot acceptance')
    write(out/'plan.json',plan);shutil.copy2(__file__,out/'source.py')
    original=evaluate.configure_samples
    for entry in plan['models']:
        def configured(env,samples,folder):
            value=original(env,samples,folder);a=env.action
            assert (a.cfg.leg_stiffness,a.cfg.leg_damping,a.cfg.wheel_stiffness)==(entry['p'],entry['d'],.1)
            assert not a.wheel_motor.enabled and not a._wheel_speed_shaping and a.motor_strength.eq(1).all()
            write(Path(folder)/'actual-control.json',dict(leg_p=a.cfg.leg_stiffness,leg_d=a.cfg.leg_damping,wheel_p=a.cfg.wheel_stiffness,
                wheel_response_enabled=a.wheel_motor.enabled,mechanical=env.wheel_physics_profile))
            return value
        args=SimpleNamespace(checkpoint=Path(entry['checkpoint']),output=out/entry['name'],deadline=time.time()+300,
            diagnostic=True,diagnostic_height=None,batch_size=9,selection=None,leg_load_profile='legacy',response_parameters=None)
        with patch.object(evaluate,'evaluation_items',items),patch.object(evaluate,'from_sim2sim_config',return_value=conditions),patch.object(evaluate,'configure_samples',configured),patch.object(FudanWalkCfg.control,'leg_stiffness',entry['p']),patch.object(FudanWalkCfg.control,'leg_damping',entry['d']),patch.object(FudanWalkCfg.control,'wheel_stiffness',.1):
            evaluate.gpu(args)
    report(out)


def report(out):
    plan=json.loads((out/'plan.json').read_text());rows=[];pairs=[]
    for entry in plan['models']:
        folder=out/entry['name'];mapping=json.loads((folder/'fixtures/mapping.json').read_text());legs=np.asarray(mapping['motor_q'])[[0,1,3,4]]
        for sample in plan['samples']:
            end=sample['case']['duration'];height=sample['case']['segments'][0]['command'][2]
            with np.load(folder/'traces'/(sample['id']+'.npz')) as tr:
                assert abs(tr['time'][-1]-end)<1e-6 and np.isfinite(tr['qpos']).all()
                last=(tr['time']>=end-5)&(tr['time']<end);one=(tr['qpos_time']>=end-1)&(tr['qpos_time']<end)
                rows.append(dict(model=entry['name'],sample_id=sample['id'],case=sample['case']['category'],height_command_m=height,
                    height_last5_mean_m=float(tr['position'][last,2].mean()),pitch_last5_peak_deg=float(abs(tr['roll_pitch_deg'][last,1]).max()),
                    speed_last5_peak_m_s=float(np.linalg.norm(tr['linear_world'][last,:2],axis=1).max()),
                    body_contact_last5_mean_N=float(tr['base_contact'][last].mean()),wheel_support_last5_mean_N=tr['wheel_support'][last].mean(0).tolist(),
                    wheel_loaded_last5_fraction=float(np.mean(np.all(tr['wheel_support'][last]>.1*tr['weight'][last,None],axis=1))),
                    leg_last1_max_pp_deg=float(np.ptp(np.rad2deg(tr['qpos'][one][:,legs]),axis=0).max())))
    for sample in plan['samples']:
        path='fixtures/'+sample['id']+'_initial.npz'
        with np.load(out/plan['models'][0]['name']/path) as left,np.load(out/plan['models'][1]['name']/path) as right:
            equal=all(np.array_equal(left[k],right[k]) for k in left.files)
            assert equal,sample['id'];pairs.append(dict(sample_id=sample['id'],all_fixture_fields_equal=equal))
    write(out/'summary.json',dict(rows=rows,paired_initial_states=pairs,training_updates=0,behavior_accepted=False))
    print(json.dumps(rows,indent=2))


def render(out):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    plan=json.loads((out/'plan.json').read_text());video=out/'video';video.mkdir(exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);videos=[]
    for category in ('zero','stop_forward','stop_backward'):
        sample=next(s for s in plan['samples'] if s['case']['category']==category and s['case']['segments'][0]['command'][2]==.26)
        end=sample['case']['duration'];sides=[]
        for entry in plan['models']:
            folder=out/entry['name'];model=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth=640;model.vis.global_.offheight=480;data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.distance=1.5;camera.elevation=-12.;camera.azimuth=90.
            option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
            with np.load(folder/'traces'/(sample['id']+'.npz')) as f:tr={k:f[k].copy() for k in f.files}
            mapping=json.loads((folder/'fixtures/mapping.json').read_text());legs=np.asarray(mapping['motor_q'])[[0,1,3,4]]
            sides.append((entry,model,data,camera,option,tr,legs,mujoco.Renderer(model,width=640,height=480)))
        target=video/(category+'.mp4');frames=0
        try:
            with imageio.get_writer(target,fps=50,codec='libx264',quality=8) as writer:
                for k in range(0,round(end*100),2):
                    t=k*.01;tick=k*5;command=evaluate.command(sample['case'],t);canvas=Image.new('RGB',(1280,544),'#18212b');draw=ImageDraw.Draw(canvas)
                    draw.text((12,8),f'1x NORMAL | {category} | t={t:.2f}s | cmd vx/yaw/h={command} | no external force',font=font,fill='white')
                    draw.text((12,34),'Left: source3000 P60/D2 | Right: continuation3500 P20/D1 | wheel P0.1; original mechanical parameters',font=font,fill='#ffcd72')
                    for col,(entry,model,data,cam,option,tr,legs,renderer) in enumerate(sides):
                        data.qpos[:]=tr['qpos'][k];mujoco.mj_forward(model,data);cam.lookat[:]=[data.qpos[0],data.qpos[1],.20]
                        renderer.update_scene(data,camera=cam,scene_option=option);panel=Image.fromarray(renderer.render().copy());d=ImageDraw.Draw(panel);d.rectangle((0,0,640,112),fill='#18212b')
                        pp=np.ptp(np.rad2deg(tr['qpos'][max(0,k-100):k+1][:,legs]),axis=0).max();speed=np.linalg.norm(tr['linear_world'][tick,:2]);support=tr['wheel_support'][tick]
                        lines=[entry['name'],f'Actual h={tr["position"][tick,2]:.3f}m | pitch={tr["roll_pitch_deg"][tick,1]:+.1f}deg | speed={speed:.3f}m/s',
                            f'Wheel L/R={support[0]:.1f}/{support[1]:.1f}N | Body={tr["base_contact"][tick]:.1f}N',f'Leg last1s p-p={pp:.2f}deg | XY following camera']
                        for row,text in enumerate(lines):d.text((10,7+25*row),text,font=font,fill='white')
                        canvas.paste(panel,(col*640,64))
                    frame=np.asarray(canvas);writer.append_data(frame);frames+=1
                    if k==round((end-1)*100):imageio.imwrite(video/(category+'.png'),frame)
        finally:
            for side in sides:side[-1].close()
        reader=imageio.get_reader(target);meta=reader.get_meta_data();decoded=sum(1 for _ in reader);reader.close();assert decoded==frames==round(end*50) and meta['fps']==50
        videos.append(dict(file=target.name,frames=frames,fps=50,duration_s=end,sha256=sha(target),full_decode_passed=True))
    write(video/'index.json',dict(videos=videos,new_physics_steps=0,policy_forwards=0))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','report','render'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    {'run':run,'report':report,'render':render}[a.mode](a.output.resolve())


if __name__=='__main__':main()
