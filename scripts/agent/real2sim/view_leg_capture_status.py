"""Display an existing USB recording in MuJoCo without owning USB or sending commands."""
import argparse
import json
from pathlib import Path
import signal
import time

import mujoco
import numpy as np

from scripts.agent.real2sim.inspect_live_leg_capture import inspect
from scripts.agent.real2sim.view_h7_joint_limits import CONFIG, solve
from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--viewer-port',type=int,default=8778)
    parser.add_argument('--point-status',type=Path)
    args=parser.parse_args()
    metadata=json.loads((args.session/'meta.json').read_text())
    step_mode=metadata.get('firmware_manifest',{}).get('expected_build',{}).get('step_mm')==10
    total_ms=metadata.get('firmware_manifest',{}).get('expected_build',{}).get('joint_capture_total_ms',360000)
    import viser
    from mjlab.viewer.viser.scene import MjlabViserScene
    pose=replay.mirror.MirrorPose(machine_config=CONFIG)
    pose.data.qpos[:7]=(0.,0.,.8,1.,0.,0.,0.)
    server=viser.ViserServer(host='127.0.0.1',port=args.viewer_port,label='MuJoCo · 腿长往返采集')
    scene=MjlabViserScene(server,pose.model,1)
    scene.create_scene_gui(camera_distance=1.5,camera_azimuth=110.,camera_elevation=15.,show_debug_viz_control=False)
    server.gui.add_markdown('**USB实录实时显示；页面不驱动机器人。**\n\n左右按机器人自身。腿长为髋轴到轮轴的解算距离；力矩为电机驱动器反馈。')
    status=server.gui.add_text('反馈状态',initial_value='等待记录',disabled=True)
    table=server.gui.add_markdown('等待数据…')
    sample_status=server.gui.add_text('静态取点',initial_value='等待稳定5秒',disabled=True) if args.point_status else None
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(f'http://127.0.0.1:{args.viewer_port}',flush=True)
    try:
        while not stopped:
            try:
                result=inspect(args.session)
                age=time.time()-(args.session/'stream.jsonl').stat().st_mtime
                values=np.zeros(8);values[::2]=result['motor_positions_rad']
                joints=solve(pose,values)
                if joints is not None:
                    scene.update_from_arrays(pose.data.xpos[None],pose.data.xmat.reshape(1,pose.model.nbody,3,3),
                                             qpos=pose.data.qpos[None],qvel=pose.data.qvel[None],ctrl=pose.data.ctrl[None])
                status.value=f"{'USB更新' if age<.5 else '记录停更'} | 策略{result['strategy']} | 故障{result['fault']} | 板时{result['board_s']:.2f}s"
                targets=result['targets_m']
                torques=result['motor_feedback_torques_nm']
                flags=result['geometry_flags']
                phase=('保护已锁止，须左下退出' if flags&16 else '仅收腿恢复中' if flags&32
                       else '边界返回控制：仍可退回' if flags&64 and any(length<=.14 or length>=.33 for length in result['lengths_m'])
                       else '正常双向段' if flags&8 else '失能 / 未运行腿长控制')
                table.content=('| 项目 | 左腿 | 右腿 |\n|---|---:|---:|\n'
                    f"| 当前长度 mm | {result['lengths_m'][0]*1000:.1f} | {result['lengths_m'][1]*1000:.1f} |\n"
                    +(f'| 目标长度 mm | {targets[0]*1000:.1f} | {targets[1]*1000:.1f} |\n' if targets else '')
                    +f'| 前电机反馈力矩 N·m | {torques[0]:.3f} | {torques[2]:.3f} |\n'
                    +f'| 后电机反馈力矩 N·m | {torques[1]:.3f} | {torques[3]:.3f} |\n'
                    +('\n腿测试拨轮：**每次10 mm，回中后再拨；目标平滑到位**；' if flags&128 or step_mode
                      else f"\n拨轮实收速率：**{result['dial_rate_m_s']*1000:.1f} mm/s**；")
                    +phase
                    +'。机身固定显示，无物理积分。')
                capture=result.get('joint_capture')
                if capture is not None:
                    labels=('左前','左后','右前','右后')
                    stages=('预检扫频','正式采集','验证扫频','完成保持')
                    state='保护停止，左下退出' if capture['flags']&2 else stages[min(capture['stage'],3)]
                    duration=120. if capture['flags']&4 else total_ms/1000.
                    if capture['flags']&4 and capture['stage']!=3:state='高频小幅验证（±0.05rad，2→5Hz）'
                    if capture['flags']&24:
                        from scripts.agent.real2sim.fit_leg_length_response import leg_pose
                        duration=25.
                        state='保护停止，左下退出' if capture['flags']&2 else ('接管保持','接管保持','右腿缓慢定位','右腿终点保持')[min(capture['stage'],3)]
                        goal_length,goal_angle,_=leg_pose(np.asarray(capture['target_firmware_rad'])[None])
                    table.content=(f"**{state} · {labels[min(capture['axis'],3)]} · {capture['elapsed_ms']/1000:.1f}/{duration:.0f} s**\n\n"
                        '| 关节 | 目标 rad | 实测 rad | 反馈 N·m |\n|---|---:|---:|---:|\n'
                        +''.join(f"| {name} | {capture['target_firmware_rad'][j]:.4f} | {result['motor_positions_rad'][j]:.4f} | {torques[j]:.3f} |\n" for j,name in enumerate(labels))
                        +f"\n腿长：左{result['lengths_m'][0]*1000:.1f} / 右{result['lengths_m'][1]*1000:.1f} mm。左下退出；当前显示仅接收。")
                    if capture['flags']&24:
                        goal_description='最终190mm，腿角锁定接管实测值' if capture['flags']&16 else '最终190mm/30°'
                        table.content+=f"\n\n**右腿长度：当前{result['lengths_m'][1]*1000:.2f} / 目标{goal_length[0,1]*1000:.2f} mm**\n\n**右腿角度：当前{np.degrees(result['angles_rad'][1]):.2f} / 目标{np.degrees(goal_angle[0,1]):.2f}°**\n\n{goal_description}；左腿及双轮请求为零，左腿处于自由状态。"
                if sample_status is not None and args.point_status.is_file():
                    progress=json.loads(args.point_status.read_text())
                    saved=[]
                    for side,label in enumerate(('left','right')):
                        matches=[p for p in progress['points'] if p['side']==label and targets is not None
                                 and p.get('control_targets_m') is not None
                                 and abs(p['control_targets_m'][side]-targets[side])<.00002
                                 and p['direction']==progress.get('direction')]
                        saved.append(matches[-1] if matches else None)
                    if age>=.5 or time.time()-progress['updated_unix_s']>2:
                        sample_status.value='记录停更，暂不换档'
                    elif result['dial_rate_m_s']!=0:
                        sample_status.value='拨轮未回中，暂不取点'
                    elif all(saved):
                        sample_status.value=f"当前档已记录：左{saved[0]['actual_length_mm']:.1f} / 右{saved[1]['actual_length_mm']:.1f} mm"
                    else:
                        sample_status.value='等待稳定5秒：左'+('已记' if saved[0] else '等待')+' / 右'+('已记' if saved[1] else '等待')
            except (ValueError,IndexError) as error:
                status.value='等待完整记录：'+str(error)
            time.sleep(.1)
    finally:server.stop()


if __name__=='__main__':main()
