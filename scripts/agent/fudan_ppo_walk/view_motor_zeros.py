"""Static MuJoCo motor-zero direction overlay; never integrate an unclosed zero pose."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import viser
from mjlab.viewer.viser.scene import MjlabViserScene
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.reset_geometry import ResetGeometry
from ct_mjlab.fudan_ppo_walk.contract import LEGS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=8776)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cfg=FudanWalkCfg();solver=ResetGeometry();model=solver.model
    nominal=np.zeros(6,dtype=np.float32);nominal[list(LEGS)]=cfg.control.nominal_leg_positions
    # 同时四零未求得合法闭链时，不强制拼出一幅假站姿。
    try:
        zero_pose=solver.solve(np.zeros(6,dtype=np.float32))
        zero_search=dict(passed=True,residual_m=zero_pose['closed_chain_residual_m'])
    except ValueError as error:
        zero_search=dict(passed=False,error=str(error))
    pose=solver.solve(nominal)
    data=mujoco.MjData(model);data.qpos[:7]=(0.,0.,pose['height'],1.,0.,0.,0.)
    for name,value in pose['joint_positions'].items():data.qpos[solver.qaddr[name]]=value
    mujoco.mj_forward(model,data)
    specs=[('lf0','lf1',-1,(255,150,35)),('lf00','lf01',1,(20,205,240)),
           ('rf0','rf1',-1,(255,150,35)),('rf00','rf01',1,(20,205,240))]
    arrows=[]
    for name,child,sign,color in specs:
        # 只对该关节做静态运动学计算，提取q=0时杆方向的XZ投影。
        probe=mujoco.MjData(model);probe.qpos[:]=data.qpos;probe.qpos[solver.qaddr[name]]=0.
        mujoco.mj_kinematics(model,probe)
        parent=model.body(name).id;tip=model.body(child).id
        direction=probe.xpos[tip]-probe.xpos[parent];direction[1]=0.;direction/=np.linalg.norm(direction)
        assert np.allclose(direction,[sign,0.,0.],atol=1e-7),(name,direction)
        anchor=data.xanchor[model.joint(name).id].copy();outside=anchor.copy()
        outside[1]+=.16 if name.startswith('lf') else -.16
        end=outside+direction*(.24 if sign<0 else .17)
        arrows.append(dict(name=name,axis_position=anchor.tolist(),display_origin=outside.tolist(),
            zero_direction_xz=direction.tolist(),end=end.tolist(),color=color,
            current_q_rad=float(data.qpos[solver.qaddr[name]])))
    report=dict(kind='static_motor_zero_directions',all_four_zero_search=zero_search,arrows=arrows,
        displayed_pose='legal_default_pose_with_zero_direction_overlays',physics_steps=0,policy_forwards=0,
        closed_chain_residual_m=pose['closed_chain_residual_m'],viewer_url=f'http://127.0.0.1:{args.port}')
    (args.output/'zero-directions.json').write_text(json.dumps(report,indent=2)+'\n')
    server=viser.ViserServer(host='127.0.0.1',port=args.port,label='MuJoCo · 四电机零位方向')
    server.gui.configure_theme(dark_mode=True)
    scene=MjlabViserScene(server,model,1)
    scene.create_scene_gui(camera_distance=1.35,camera_azimuth=110.,camera_elevation=15.,show_debug_viz_control=False)
    scene.update_from_arrays(data.xpos[None],data.xmat.reshape(1,model.nbody,3,3),qpos=data.qpos[None],qvel=data.qvel[None],ctrl=data.ctrl[None])
    overlay_root=scene.fixed_bodies_frame.name+'/motor_zero_reference'
    server.scene.add_grid(overlay_root+'/ground',width=3.,height=3.,cell_size=.1,plane='xy')
    groups={side:server.scene.add_frame(overlay_root+'/'+side,show_axes=False,visible=True) for side in ('lf','rf')}
    for item in arrows:
        name=item['name'];side=name[:2];start=np.array(item['display_origin']);end=np.array(item['end']);direction=np.array(item['zero_direction_xz'])
        head=end-direction*.035
        segments=np.array([[start,end],[end,head+[0.,0.,.02]],[end,head-[0.,0.,.02]]])
        base=overlay_root+'/'+side+'/'+name
        server.scene.add_line_segments(base+'/arrow',segments,colors=item['color'],thickness=.007)
        server.scene.add_line_segments(base+'/axis-guide',np.array([[item['axis_position'],start]]),colors=item['color'],thickness=.002)
        label=end+np.array([-.06 if direction[0]<0 else .01,0.,.04])
        server.scene.add_label(base+'/label',name+'  q=0  '+('-X 后方' if direction[0]<0 else '+X 前方'),position=label,font_screen_scale=.9,visible=False)
    with server.gui.add_folder('电机零位参考',expand_by_default=True):
        server.gui.add_markdown('**橙色：lf0 / rf0 主腿杆零位 → 后方 −X**\n\n**蓝色：lf00 / rf00 短摇臂零位 → 前方 +X**\n\n箭头是侧面XZ投影，外移显示并用细线连回轴心。机器人本体保留合法默认姿态。\n\n当前未找到四电机同时归零的合法闭链，不能将此图当成全零站姿；没有推进动力学或运行策略。')
        side_choice=server.gui.add_dropdown('查看侧别',options=('lf侧','rf侧','双侧'),initial_value='双侧')
        @side_choice.on_update
        def select_side(_):
            groups['lf'].visible=side_choice.value in ('lf侧','双侧')
            groups['rf'].visible=side_choice.value in ('rf侧','双侧')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    try:
        while True:time.sleep(1.)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

if __name__=='__main__':main()
