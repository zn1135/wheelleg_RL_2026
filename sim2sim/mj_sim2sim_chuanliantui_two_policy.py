#!/usr/bin/env python3
"""Independent CPU MuJoCo: select two no-mode ONNX policies here, never during training."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from mj_sim2sim_chuanliantui_stand_v2 import StandV2Playback, POLICY_DT
from mj_sim2sim_huananhu_multimode import sha256, DT, MODEL_ASSET
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_split.runner import validate
from ct_mjlab.envs.chuanliantui_split.observations import OBSERVATION_CONTRACT
from ct_mjlab.envs.chuanliantui_split.switch import PolicySwitch, conditions, SWITCH_CONTRACT

CONTRACT = 'chuanliantui-two-policy-native-cpu-history125-episode-height-v2'


class OnnxPolicy:
    def __init__(self, checkpoint, path, role):
        import onnx
        from onnx.reference import ReferenceEvaluator
        infos = torch.load(checkpoint, map_location='cpu', weights_only=False)['infos']
        validate(infos, command_cfg(role))
        # CPU交接固定起立0.22m，不能把尚未解锁的0.30m策略当作同一阶段运行。
        if role == 'getup' and not infos['standup_curriculum']['unlocked']:
            raise ValueError('Two-policy CPU playback requires get-up height curriculum unlocked to 0.22 m')
        if role == 'getup' and infos.get('action_filter_contract'):
            raise ValueError('Two-policy CPU playback has not implemented get-up leg action smoothing')
        record = json.loads(path.with_name(path.stem+'_export.json').read_text())
        if (record['checkpoint_sha256'] != sha256(checkpoint) or record['onnx_sha256'] != sha256(path)
                or record['policy_role'] != role or record['observation_contract'] != OBSERVATION_CONTRACT
                or record['switch_contract'] != SWITCH_CONTRACT):
            raise ValueError('ONNX role, source, graph hash or interface differs from checkpoint')
        if role == 'walk' and (record.get('command_contract') != infos['command_contract']
                or record.get('height_range') != list(infos['command_config']['height_range'])):
            raise ValueError('ONNX walking height or command contract differs from checkpoint')
        graph = onnx.load(str(path));onnx.checker.check_model(graph)
        shapes = {v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.graph.input}
        if shapes != {'observation_history':[1,125]}:
            raise ValueError(f'Unexpected two-policy ONNX input: {shapes}')
        self.evaluator = ReferenceEvaluator(graph)
        self.infos, self.role = infos, role

    def __call__(self, history):
        action, velocity = self.evaluator.run(None, {'observation_history':history.reshape(1,125)})
        if action.shape != (1,6) or velocity.shape != (1,3) or not np.isfinite(action).all() or not np.isfinite(velocity).all():
            raise FloatingPointError('Non-finite or malformed ONNX output')
        return action[0]


class CommandController:
    """Caller-owned yaw/position P loops, matching the walking command recipe."""
    def __init__(self, config, forward_speed_limit, backward_speed_limit):
        self.cfg = config
        self.forward_speed_limit, self.backward_speed_limit = forward_speed_limit, backward_speed_limit
        self.reset()

    def reset(self):
        self.vx = self.yaw = 0.
        self.heading_reference = self.anchor = None
        self.ready_ticks = self.quiet_ticks = 0
        self.forward_error = self.lateral_error = self.position_velocity = 0.

    def update(self, xy, heading, body_vx, requested_vx, requested_yaw, ready):
        c = self.cfg
        self.ready_ticks = self.ready_ticks+1 if ready else 0
        if self.ready_ticks < 50:
            ticks = self.ready_ticks
            self.reset();self.ready_ticks = ticks
            return np.array((0.,0.,.22))
        if requested_yaw:
            self.heading_reference = None
            yaw_target = np.clip(requested_yaw,-c['yaw_rate_limit'],c['yaw_rate_limit'])
        else:
            if self.heading_reference is None:self.heading_reference = heading
            error = np.arctan2(np.sin(self.heading_reference-heading),np.cos(self.heading_reference-heading))
            yaw_target = np.clip(c['heading_gain']*error,-c['yaw_rate_limit'],c['yaw_rate_limit'])
        self.yaw += np.clip(yaw_target-self.yaw,-c['yaw_acceleration']*POLICY_DT,c['yaw_acceleration']*POLICY_DT)
        target = np.clip(requested_vx,-self.backward_speed_limit,self.forward_speed_limit)
        self.forward_error = self.lateral_error = self.position_velocity = 0.
        if requested_vx:
            self.anchor = None;self.quiet_ticks = 0
        else:
            if self.anchor is None:
                self.quiet_ticks = self.quiet_ticks+1 if abs(body_vx)<c['position_lock_speed'] else 0
                if self.quiet_ticks >= int(np.ceil(c['position_lock_duration']/POLICY_DT)):
                    self.anchor = np.asarray(xy).copy()
            if self.anchor is not None:
                delta = np.asarray(xy)-self.anchor
                self.forward_error = float(delta @ np.array((np.cos(heading),np.sin(heading))))
                self.lateral_error = float(delta @ np.array((-np.sin(heading),np.cos(heading))))
                if abs(self.forward_error)>c['position_tolerance']:
                    target = np.clip(-c['position_gain']*self.forward_error,
                                     -c['position_velocity_limit'],c['position_velocity_limit'])
                self.position_velocity = float(target)
        # Caller acceleration 1 m/s² is within the trained [0.5,2] command range.
        self.vx += np.clip(target-self.vx,-POLICY_DT,POLICY_DT)
        return np.array((self.vx,self.yaw,.22))


class TwoPolicyPlayback(StandV2Playback):
    def __init__(self, initial_states, getup, walk, sample=0, vx=0., yaw=0., height=.22):
        self.policies = (getup,walk)
        self.selector = PolicySwitch(1,'cpu')
        info = walk.infos
        speed = info['standup_curriculum']['speed_curriculum']
        forward_limit = info['command_config']['forward_speed_limits'][speed['forward']['stage']]
        backward_limit = info['command_config']['backward_speed_limits'][speed['backward']['stage']]
        if not -backward_limit <= vx <= forward_limit or abs(yaw)>info['command_config']['yaw_rate_limit']:
            raise ValueError('Requested velocity exceeds this checkpoint command limits')
        low,high = info['command_config']['height_range']
        if not np.isfinite(height) or not low <= height <= high:
            raise ValueError('Requested walking height exceeds this checkpoint range')
        if vx and yaw:
            raise ValueError('Walking recipe permits straight travel or spin, not both requests')
        self.command_loop = CommandController(info['command_config'],forward_limit,backward_limit)
        self.requested_vx,self.requested_yaw = vx,yaw
        self.requested_height = height
        self.transitions = 0
        super().__init__(initial_states,sample)

    def reset(self, pose=None):
        super().reset(pose)
        self.selector.reset(torch.tensor([0]));self.command_loop.reset()
        self.command[:] = (0.,0.,.22)
        self.transitions = 0

    def observation(self):
        return super().observation()[:25]

    def update_history(self, frame):
        if self.history is None:self.history = np.tile(frame,(5,1))
        else:
            self.history[:-1] = self.history[1:].copy();self.history[-1] = frame
        return self.history.reshape(125)

    def switch_conditions(self):
        rotation = np.empty(9);mujoco.mju_quat2Mat(rotation,self.data.qpos[3:7])
        rotation = rotation.reshape(3,3)
        gravity = rotation.T @ np.array((0.,0.,-1.))
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        arrays = (gravity[2],self.data.qpos[2],np.linalg.norm(self.data.qvel[3:5]),self.base_force)
        values = [torch.tensor([float(v)],dtype=torch.float64) for v in arrays]
        return conditions(*values,torch.tensor(self.wheel_support[None]),
                          torch.tensor([self.weight]),torch.tensor([finite]))

    def update_mode(self):
        super().update_mode()
        # Falls switch policies; only the planned episode endpoint ends playback.
        self.done = self.steps>=4000

    def step(self, controller=None):
        if self.requested_vx and self.requested_yaw:
            raise ValueError('Walking recipe permits straight travel or spin, not both requests')
        ready,down,emergency,finite = self.switch_conditions()
        changed = self.selector.update(ready,down,emergency)
        if changed.any():
            self.transitions += 1;self.command_loop.reset()
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,self.data.qpos[3:7]);rotation=rotation.reshape(3,3)
        heading=np.arctan2(rotation[1,0],rotation[0,0]);body_vx=float((rotation.T@self.data.qvel[:3])[0])
        walking=bool(self.selector.walk[0])
        # selector的100ms退出滞回容忍短掉载；不再叠加getup recovered或当拍down门控。
        self.command[:] = self.command_loop.update(self.data.qpos[:2],heading,body_vx,
            self.requested_vx,self.requested_yaw,walking and self.steps >= round(1. / POLICY_DT))
        # Get-up stays at 0.22 m; the walking height is fixed for this playback run.
        if walking:self.command[2] = self.requested_height
        if not finite.all():raise FloatingPointError('Non-finite CPU state before inference')
        # Select before constructing the current frame; no physical/action/FIFO reset here.
        return super().step(self.policies[int(walking)])

    def metrics(self, action=None, torque=None):
        result=super().metrics(action,torque)
        result.update(policy_role='walk' if self.selector.walk[0] else 'getup',transitions=self.transitions,
            requested_vx=self.requested_vx,requested_yaw=self.requested_yaw,
            requested_height_m=self.requested_height,
            actual_command=self.command.tolist(),position_locked=self.command_loop.anchor is not None,
            forward_position_error_m=self.command_loop.forward_error,lateral_position_error_m=self.command_loop.lateral_error)
        return result


def main(args):
    if args.report and args.report.exists():raise FileExistsError(args.report)
    policies=[OnnxPolicy(checkpoint,graph,role) for checkpoint,graph,role in (
        (args.getup_checkpoint,args.getup_onnx,'getup'),(args.walk_checkpoint,args.walk_onnx,'walk'))]
    env=TwoPolicyPlayback(args.initial_states,*policies,sample=args.sample,vx=args.vx,yaw=args.yaw,height=args.height)
    rows=[];server=scene=None;completed=False
    if args.viewer:
        import viser
        from mjlab.viewer.viser.scene import MjlabViserScene
        server=viser.ViserServer(host='127.0.0.1',port=args.port,label='two-policy CPU sim2sim')
        scene=MjlabViserScene(server,env.model,1)
        scene.create_scene_gui(camera_distance=2.,show_debug_viz_control=False)
        paused=server.gui.add_checkbox('暂停',initial_value=False)
        restart=server.gui.add_button('重放当前初态')
        pending=[];restart.on_click(lambda _:pending.append(True))
        vx=server.gui.add_slider('前进速度请求 m/s',min=-env.command_loop.backward_speed_limit,max=env.command_loop.forward_speed_limit,
                                 step=.01,initial_value=args.vx)
        yaw=server.gui.add_slider('转向速度请求 rad/s',min=-.5,max=.5,step=.01,initial_value=args.yaw)
        @vx.on_update
        def straight(_):
            if vx.value:yaw.value=0.
        @yaw.on_update
        def spin(_):
            if yaw.value:vx.value=0.
        info=server.gui.add_markdown('起立 → 行走；倒地后切回起立。两个网络均无模式输入。')
        scene.update_from_mjdata(env.data)
        print(f'Two-policy CPU sim2sim: http://127.0.0.1:{server.get_port()}',flush=True)
    try:
        ticks=0
        row=env.metrics()
        while args.viewer or ticks<round(args.seconds/POLICY_DT):
            started=time.monotonic()
            if server and pending:
                pending.clear();env.reset()
            if server is None or not paused.value:
                if env.done:
                    if server is None:break
                    env.reset()
                if server:env.requested_vx,env.requested_yaw=vx.value,yaw.value
                action,torque=env.step();ticks+=1
                row=env.metrics(action,torque)
                if args.report:rows.append(row)
                if ticks%100==0:print(json.dumps(row),flush=True)
            if scene is not None:
                if ticks%3==0:
                    with server.atomic():
                        scene.update_from_mjdata(env.data)
                        info.content=f"**{row['policy_role']}** · {env.data.time:.1f}s · 切换{env.transitions}次"
                time.sleep(max(0.,POLICY_DT-(time.monotonic()-started)))
        completed=True
    finally:
        if server:server.stop()
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            record=dict(contract=CONTRACT,completed=completed,rows=rows,parameter_updates=0,
                observation_contract=OBSERVATION_CONTRACT,switch_contract=SWITCH_CONTRACT,
                checkpoint_sha256=[sha256(args.getup_checkpoint),sha256(args.walk_checkpoint)],
                onnx_sha256=[sha256(args.getup_onnx),sha256(args.walk_onnx)],
                xml_sha256=sha256(MODEL_ASSET),source_sha256=sha256(Path(__file__)),
                policy_dt_s=POLICY_DT,pd_dt_s=DT,integration_dt_s=DT/4,
                scope='Independent CPU MuJoCo and two ONNX policies; no training env physics; H7 unmodified')
            with args.report.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('getup-checkpoint','getup-onnx','walk-checkpoint','walk-onnx','initial-states'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sample',type=int,default=0)
    parser.add_argument('--seconds',type=float,default=40.)
    parser.add_argument('--vx',type=float,default=0.)
    parser.add_argument('--yaw',type=float,default=0.)
    parser.add_argument('--height',type=float,default=.22,help='Fixed walking height for this run, within trained range')
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--port',type=int,default=8094)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.seconds<=0:parser.error('--seconds must be positive')
    main(args)
