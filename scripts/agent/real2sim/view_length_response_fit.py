"""Saved right-leg measurement versus GPU response; no USB or control interface."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import viser
from mjlab.viewer.viser.scene import MjlabViserScene


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8780)
    args=parser.parse_args()
    proof=json.loads((args.run/'verification.json').read_text());assert proof['software_verified']
    report=json.loads((args.run/'report.json').read_text())
    with np.load(args.run/'comparison.npz') as source:a={k:source[k] for k in source.files}
    model=mujoco.MjModel.from_xml_path(str(args.run/'model.xml'))
    for i in range(model.ngeom):
        name=model.body(model.geom_bodyid[i]).name
        if name.startswith(('lf','left_')):model.geom_matid[i]=-1;model.geom_rgba[i,3]=0
        elif name=='base_link':model.geom_matid[i]=-1;model.geom_rgba[i]=[.6,.65,.72,.25]
    server=viser.ViserServer(host='127.0.0.1',port=args.port,label='右腿 · 实录与GPU仿真对照')
    scene=MjlabViserScene(server,model,2);scene.camera_tracking_enabled=False
    scene.create_scene_gui(camera_distance=2.2,camera_azimuth=270.,camera_elevation=10.,show_debug_viz_control=False)
    server.scene.add_grid('/floor',width=4.,height=4.,cell_size=.1,section_size=1.,plane='xy')
    server.gui.add_markdown('**左：实录关节角重建　右：GPU自由仿真**\n\n仅显示机器人自身右腿；机身固定。页面无USB连接，不驱动实机。')
    torque_label='同驱动器估计力矩' if report.get('torque_input_source')=='driver_estimate' else '同实录请求力矩'
    options={torque_label:1,'同目标、同控制器':2}
    mode=server.gui.add_dropdown('回放分支',options=list(options),initial_value=torque_label)
    playing=server.gui.add_checkbox('播放',initial_value=False)
    slider=server.gui.add_slider('时间 / 秒',min=0.,max=float(a['time_s'][-1]),step=.02,initial_value=0.)
    restart=server.gui.add_button('从头播放');table=server.gui.add_markdown('')
    server.gui.add_markdown('当前是这条轨迹的局部候选，**尚未消除全部响应差异**；未改固件、控制增益或训练模型。')
    @restart.on_click
    def reset(_):slider.value=0.;playing.value=True
    @server.on_client_connect
    def connect(client):
        client.camera.position=(.15,-2.3,1.05);client.camera.look_at=(.15,0.,.65);client.camera.up_direction=(0.,0.,1.)
    print(json.dumps(dict(url=f'http://127.0.0.1:{server.get_port()}/',hardware_access=False,new_physics_steps=0)),flush=True)
    last=time.monotonic()
    while True:
        now=time.monotonic()
        if playing.value and server.get_clients():
            slider.value=min(slider.max,slider.value+min(now-last,.1))
            if slider.value>=slider.max:playing.value=False
        last=now;i=min(round(slider.value/.02),len(a['time_s'])-1);branch=options[mode.value]
        positions=a['xpos'][i,[0,branch]].copy();positions[0,:,0]-=.45;positions[1,:,0]+=.45
        scene.update_from_arrays(positions,a['xmat'][i,[0,branch]],qpos=a['qpos'][i,[0,branch]],qvel=np.zeros((2,model.nv)),ctrl=np.zeros((2,model.nu)))
        length=a['length_m'][i,[0,branch]]*1000;angle=np.degrees(a['angle_rad'][i,[0,branch]]);tau=a['request_nm'][i,[0,branch]]
        m=proof['metrics'][branch-1]
        table.content='\n'.join(['| 右腿 | 实录 | 仿真 |','|---|---:|---:|',f'| 长度 mm | {length[0]:.2f} | {length[1]:.2f} |',
                                 f'| 目标长度 mm | {a["target_length_m"][i]*1000:.2f} | {a["target_length_m"][i]*1000:.2f} |',
                                 f'| 腿角 ° | {angle[0]:.2f} | {angle[1]:.2f} |',f'| 前输入 N·m | {tau[0,0]:.2f} | {tau[1,0]:.2f} |',
                                 f'| 后输入 N·m | {tau[0,1]:.2f} | {tau[1,1]:.2f} |','',f'全段长度RMSE {m["length_rmse_mm"]:.2f} mm；腿角RMSE {m["angle_rmse_deg"]:.2f}°。'])
        time.sleep(.03)


if __name__=='__main__':main()
