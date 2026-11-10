"""Display the configured default motor angles as a static MuJoCo pose; no integration."""
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
from ct_mjlab.fudan_ppo_walk.contract import MOTORS, LEGS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', type=int, default=8775)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = FudanWalkCfg()
    nominal = np.zeros(6, dtype=np.float32)
    nominal[list(LEGS)] = cfg.control.nominal_leg_positions
    solver = ResetGeometry()
    pose = solver.solve(nominal)
    model = solver.model
    data = mujoco.MjData(model)
    data.qpos[:7] = (0., 0., pose['height'], 1., 0., 0., 0.)
    for name, value in pose['joint_positions'].items():
        data.qpos[solver.qaddr[name]] = value
    # 仅计算几何位置并显示，整个入口没有mj_step、策略推理或优化器更新。
    mujoco.mj_forward(model, data)
    actual = {name: float(data.qpos[solver.qaddr[name]]) for name in ('lf0', 'lf00', 'rf0', 'rf00')}
    report = dict(kind='static_configured_default_pose',default_motor_angles_rad=actual,
        configured_angles_rad=dict(zip(('lf0','lf00','rf0','rf00'),cfg.control.nominal_leg_positions)),
        height_m=pose['height'],closed_chain_residual_m=pose['closed_chain_residual_m'],
        minimum_clearance_m=pose['minimum_ground_clearance_m'],random_offsets=False,
        physics_steps=0,policy_forwards=0,optimizer_updates=0)
    (args.output/'pose.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'pose.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy())
    model.vis.global_.offwidth=960
    model.vis.global_.offheight=640
    option=mujoco.MjvOption()
    option.geomgroup[:]=[1,1,1,0,0,0]
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0.,0.,.19]
    camera.distance=1.15
    camera.elevation=-12.
    with mujoco.Renderer(model,height=640,width=960) as renderer:
        from PIL import Image
        for name,azimuth in [('side',90.),('front',0.),('perspective',125.)]:
            camera.azimuth=azimuth
            renderer.update_scene(data,camera=camera,scene_option=option)
            Image.fromarray(renderer.render()).save(args.output/(name+'.png'))
    server=viser.ViserServer(host='127.0.0.1',port=args.port,label='复旦默认角 · MuJoCo静态姿态')
    scene=MjlabViserScene(server,model,1)
    scene.create_scene_gui(camera_distance=1.25,camera_azimuth=90.,camera_elevation=15.,show_debug_viz_control=False)
    scene.update_from_arrays(data.xpos[None],data.xmat.reshape(1,model.nbody,3,3),qpos=data.qpos[None],qvel=data.qvel[None],ctrl=data.ctrl[None])
    server.scene.add_grid('/ground_reference',width=3.,height=3.,cell_size=.1,plane='xy')
    with server.gui.add_folder('当前默认角与姿态',expand_by_default=True):
        server.gui.add_markdown('**MuJoCo默认角静态姿态**\n\n可拖动旋转查看；显示基准角，未叠加随机偏置。\n\n'+
            '\n'.join(f'- **{name}**：{value:+.8f} rad（{np.degrees(value):+.2f}°）' for name,value in actual.items())+
            f'\n\n机身高度：**{pose["height"]:.4f} m**；轮底间隙约2mm。\n\n固定姿态展示，不是策略运行回放。')
    print(json.dumps(dict(url=f'http://127.0.0.1:{server.get_port()}',**report),ensure_ascii=False),flush=True)
    try:
        while True:
            time.sleep(1.)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

if __name__=='__main__':
    main()
