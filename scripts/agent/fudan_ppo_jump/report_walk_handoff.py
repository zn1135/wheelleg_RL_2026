"""实机和GPU配对轨迹的俯仰、轮矩、轮速与姿态慢放。"""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from .reproduce_walk_handoff import ARCHIVE,read_csv,matrix,write_json,mesh_vertices


def run(folder):
    cap=json.loads((folder/'capture.json').read_text());raw=np.load(folder/'gpu.npz');cases=cap['cases'];t0=cap['real_walk_s']
    with zipfile.ZipFile(ARCHIVE) as z:s,p=[read_csv(z,'试次2-'+n+'.csv') for n in ('H7反馈','H7控制')]
    real_pitch=Rotation.from_quat(matrix(s,[f'quat_{k}' for k in [1,2,3,0]])).as_euler('xyz',degrees=True)[:,1];real_t=(s['rel_s']-t0)*1000;pd_t=(p['rel_s']-t0)*1000
    real_speed=matrix(p,['current_4','current_5'])*[.0525,-.0525];real_torque=matrix(p,['pd_Nm_4','pd_Nm_5'])*[1,-1]
    model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));data=mujoco.MjData(model)
    wheel_dofs=[model.joint('robot/'+n).dofadr[0] for n in ('lfwheel','rfwheel')]
    rows=[];overview={};tracks=[]
    for j,case in enumerate(cases):
        time=np.arange(len(raw['q']))*10+case['offset_ms'];ms=np.arange(len(raw['trace']))+case['offset_ms'];pose=raw['q'][:,j];vel=raw['v'][:,j];tr=raw['trace'][:,j];ang=Rotation.from_quat(pose[:,[4,5,6,3]]).as_euler('xyz',degrees=True);pitch=ang[:,1];rate=vel[:,4]*np.cos(np.radians(ang[:,0]))-vel[:,5]*np.sin(np.radians(ang[:,0]));ws=vel[:,wheel_dofs]*[.0525,-.0525];tau=raw['torque'][:,j][:,[2,5]]*[1,-1];tracks.append((time,pitch,rate,ws,tau));m=(time>=0)&(time<=250);real=np.interp(time[m],real_t,real_pitch);idx=np.argmax(pitch[m]);row=dict(**case,switch_ms=float(raw['switch_ms'][j]),pitch_at_walk0_deg=float(pitch[time==0][0]),pitch50_deg=float(pitch[time==50][0]),pitch100_deg=float(pitch[time==100][0]),pitch200_deg=float(pitch[time==200][0]),pitch250_deg=float(pitch[time==250][0]),peak250_deg=float(pitch[m].max()),peak250_at_ms=int(time[m][idx]),absolute_peak250_deg=float(abs(pitch[m]).max()),rmse250_deg=float(np.sqrt(np.mean((pitch[m]-real)**2))),nonwheel_first250_ms=int((tr[(ms>=0)&(ms<250),10]>1).sum()),both_wheels_first250_ms=int((tr[(ms>=0)&(ms<250),8:10]>.5).all(1).sum()),terminated=bool(np.isfinite(raw['done_ms'][j])))
        rows.append(row)
        if case['scenario']=='imu' and case['offset_ms']==0:overview[case['arm']]=row
    with (folder/'cases.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for start in (-50,0):
        for name in set(c['scenario'] for c in cases):
            a=[i for i,c in enumerate(cases) if c['offset_ms']==start and c['scenario']==name];assert all(np.array_equal(raw['q0'][a[0]],raw['q0'][k]) and np.array_equal(raw['v0'][a[0]],raw['v0'][k]) and np.array_equal(raw['initial_obs'][a[0]],raw['initial_obs'][k]) for k in a)
    # 取交接时刻初始化的名义条件，用同一时间轴与实机对照。
    index={c['arm']:j for j,c in enumerate(cases) if c['offset_ms']==0 and c['scenario']=='imu'}
    labels={'walk':'原时刻切 WALK','wait':'等待组','recorded':'实机动作回放'}
    plt.rcParams['font.family']=['Noto Sans CJK SC'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    mask=(real_t>=0)&(real_t<=500);axes[0].plot(real_t[mask],real_pitch[mask],color='black',label='实机');axes[1].plot(real_t[mask],np.gradient(real_pitch[mask],real_t[mask]/1000),color='black',label='实机姿态差分')
    mask=(pd_t>=0)&(pd_t<=500);axes[2].plot(pd_t[mask],real_speed[mask].mean(1),color='black',label='实机')
    for arm,j in index.items():
        time,pitch,rate,ws,tau=tracks[j];axes[0].plot(time,pitch,label=labels[arm]);axes[1].plot(time,np.degrees(rate),label=labels[arm]);axes[2].plot(time,ws.mean(1),label=labels[arm])
    for a,label in zip(axes,['俯仰 / °','俯仰速度 / °/s','左右轮平均轮缘速度 / m/s']):a.set_ylabel(label);a.grid(alpha=.2);a.legend(ncol=4,fontsize=9);a.set_xlim(0,500);a.axvline(raw['switch_ms'][index['wait']],color='grey',ls=':')
    axes[-1].set_xlabel('相对实机切回 WALK / ms');fig.savefig(folder/'pitch.png',dpi=140);plt.close(fig)
    ar=np.load(ARCHIVE.parent/'displayed-poses.npz');sel=ar['case']==1;qt=(ar['board_us'][sel]-cap_board(s,t0))/1000;real_q=ar['qpos'][sel];bids=[model.body('robot/'+n).id for n in ('lfwheel','rfwheel')];geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g] in bids and model.geom_contype[g]]
    camera=mujoco.MjvCamera();camera.distance=1.05;camera.elevation=-8.;camera.azimuth=90+Rotation.from_quat(raw['q0'][index['walk'],[4,5,6,3]]).as_euler('xyz',degrees=True)[2];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19)
    with mujoco.Renderer(model,width=352,height=288) as renderer,imageio.get_writer(folder/'handoff-slow.mp4',fps=25,codec='libx264',quality=8) as writer:
        for k in range(51):
            canvas=Image.new('RGB',(1056,400),'#192737');draw=ImageDraw.Draw(canvas)
            for col,tag in enumerate(('real','walk','wait')):
                if tag=='real':
                    q=real_q[np.argmin(abs(qt-k*10))].copy();q[:3]=0;data.qpos[:]=q;mujoco.mj_kinematics(model,data);q[2]=-min(mesh_vertices(model,data,g)[:,2].min() for g in geoms);lab='实机姿态 · 轮底对地示意';pitch=np.interp(k*10,real_t,real_pitch)
                else:
                    j=index[tag];q=raw['q'][k,j];lab='仿真 · '+labels[tag];pitch=tracks[j][1][k]
                data.qpos[:]=q;mujoco.mj_forward(model,data);camera.lookat[:]=[q[0],q[1],.19];renderer.update_scene(data,camera=camera);canvas.paste(Image.fromarray(renderer.render()),(352*col,48));draw.text((352*col+8,10),lab,font=font,fill='white');draw.text((352*col+8,344),f'俯仰 {pitch:+.1f}°',font=font,fill='white')
            draw.text((8,373),f'交接后 {10*k} ms    0.25×    左栏为实测姿态重建，水平位移未测',font=font,fill='white');writer.append_data(np.asarray(canvas))
    with imageio.get_reader(folder/'handoff-slow.mp4') as reader:assert sum(1 for _ in reader)==51 and reader.get_meta_data()['fps']==25
    real_values={str(k):float(np.interp(k,real_t,real_pitch)) for k in (0,50,100,200,250)}
    real_values['peak_first250_deg']=float(real_pitch[(real_t>=0)&(real_t<=250)].max());real_values['peak_first300_deg']=float(real_pitch[(real_t>=0)&(real_t<=300)].max())
    write_json(folder/'report.json',dict(real=real_values,nominal=overview,all_cases=rows,paired_initial_states_exact=True,video_frames=51))
    print(json.dumps(dict(real=real_values,nominal=overview),indent=2))


def cap_board(s, switch_s):
    return np.median(s['board_us']-s['rel_s']*1e6)+switch_s*1e6


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);run(p.parse_args().folder)
