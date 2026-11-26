"""从同一空中状态比较右轮附加惯量，使用同一段已记录的动作。"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import jump_config
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from .analyze_real022 import read_csv
from .compare_landing_lead import ARCHIVE
from .probe_takeoff_work import ROOT, sha, write_json

REFERENCE = ROOT/'artifacts/fudan_ppo_jump/20261125-flight-wheel-02'
START = 18
STOP = 52
SCALES = (1., .65, .5, .4)


@torch.inference_mode()
def run(out, *, check_speed_taper=False):
    out.mkdir(parents=True, exist_ok=False)
    capture = json.loads((REFERENCE/'capture.json').read_text())
    assert capture['archive_sha']==sha(ARCHIVE)
    idx = next(i for i,c in enumerate(capture['cases']) if c['arm']=='right_wheel_log' and c['scenario']=='imu')
    with np.load(REFERENCE/'gpu.npz') as raw:
        assert raw['intervention'][START:STOP,idx].all()
        q0 = raw['q'][START,idx].copy()
        v0 = raw['v'][START,idx].copy()
        actions = raw['action'][START:STOP,idx].copy()
    with zipfile.ZipFile(ARCHIVE) as archive:
        real = read_csv(archive,'试次2-H7控制.csv')
    scales = (*SCALES, 1., .5) if check_speed_taper else SCALES
    env = JumpEnv(len(scales),823,'cuda:0',False,cfg=jump_config(),record_flat=True)
    try:
        device = env.device
        env.reset_idx(torch.arange(len(scales),device=device))
        sim = env.env.sim
        right = int(sim.mj_model.joint('robot/rfwheel').dofadr[0])
        act_id = int(sim.mj_model.actuator('robot/rfwheel_motor').id)
        assert sim.mj_model.actuator_trnid[act_id,0] == sim.mj_model.joint('robot/rfwheel').id
        assert sim.mj_model.actuator_gainprm[act_id,0] == sim.mj_model.actuator_gear[act_id,0] == 1
        assert not env.action._wheel_speed_shaping and not env.action.wheel_motor.enabled
        sim.expand_model_fields(('dof_armature',))
        armature = float(sim.model.dof_armature[0,right])
        sim.model.dof_armature[:,right] *= torch.tensor(scales,device=device)
        sim.recompute_constants(RecomputeLevel.set_const)
        sim.data.qpos[:] = torch.as_tensor(q0,device=device)
        sim.data.qvel[:] = torch.as_tensor(v0,device=device)
        sim.data.ctrl.zero_()
        sim.data.qacc_warmstart.zero_()
        sim.forward()
        env.env.scene.update(dt=0.)
        sim.sense()
        env.commands[:] = torch.tensor([0.,0.,.22],device=device)
        taper_parameters = None
        if check_speed_taper:
            from ct_mjlab.envs.chuanliantui.huananhu_action import wheel_response_input
            from ct_mjlab.envs.chuanliantui.model import load_model
            legacy_model = load_model()
            legacy_motor = legacy_model.actuator('rfwheel_motor').id
            _, cutoff, exponent = legacy_model.actuator_gainprm[legacy_motor,:3]
            # 保留原速度衰减形状，采用单位低速增益和原命令限矩。
            taper_parameters = [1.,float(cutoff),float(exponent)]
            taper = torch.tensor(taper_parameters,device=device,dtype=torch.float32)[None]
            a = env.action
            apply = a.apply_actions
            joint_ids = a._physical_joint_ids[a._wheel_action_ids[-1:]]
            def apply_taper():
                apply()
                request = a.torques[:,5:6]
                output = request.clone()
                output[-2:] = wheel_response_input(request[-2:],sim.data.qvel[-2:,right:right+1],taper)
                a._entity.set_joint_effort_target(output,joint_ids=joint_ids)
            a.apply_actions = apply_taper
        # 记录实际执行器扭矩，不混用控制器目标或实机请求字段。
        frame = env.trace_frame
        torque = []
        def capture_frame():
            trace = frame()
            torque.append(sim.data.actuator_force[:,act_id].clone())
            return trace
        env.trace_frame = capture_frame
        speeds = [sim.data.qvel[:,right].cpu().numpy().copy()]
        traces = []
        for action in torch.as_tensor(actions,device=device):
            env.substeps.clear()
            env.step(action.expand(len(scales),-1),auto_reset=False,resample=False)
            assert len(env.substeps)==10 and not sim.wp_data.overflow.numpy().any()
            speeds.append(sim.data.qvel[:,right].cpu().numpy().copy())
            traces.extend(torch.stack(env.substeps).cpu().numpy())
        speeds, traces = np.asarray(speeds), np.asarray(traces)
        torque = torch.stack(torque).cpu().numpy()
        assert np.isfinite(speeds).all() and abs(torque).max()<=3.900001
        assert not (traces[:,:,8:10]>.5).any() and not (traces[:,:,10]>1).any()
        np.testing.assert_allclose(sim.model.dof_armature[:,right].cpu().numpy(),armature*np.array(scales),atol=1e-8)
        t = np.arange(START,STOP+1)*.01
        measured = np.interp(t,real['rel_s'],real['velocity_5'])
        requested = np.interp(t,real['rel_s'],real['pd_Nm_5'])
        # 实机两端按时间插值。仿真同初始轮速，只比较相对轮速变化，单列绝对轮速。
        end_300 = 30-START
        shift = measured-measured[0]
        rows = []
        for j,multiplier in enumerate(scales):
            change = speeds[:,j]-speeds[0,j]
            rows.append(dict(scale=multiplier,armature_kg_m2=armature*multiplier,speed_taper=j>=len(SCALES),
                wheel_300=float(speeds[end_300,j]),delta_180_300=float(change[end_300]),wheel_520=float(speeds[-1,j]),
                initial_aligned_rmse_to_300=float(np.sqrt(np.mean((change[:end_300+1]-shift[:end_300+1])**2))),
                actual_rmse_330_520=float(np.sqrt(np.mean((speeds[15:,j]-measured[15:])**2))),
                torque_180_300=float(torque[:120,j].mean())))
        report = dict(archive_sha=sha(ARCHIVE),reference=str(REFERENCE.relative_to(ROOT)),
            window_ms=[180,520],real_v180=float(measured[0]),real_v300=float(measured[end_300]),
            real_v520=float(measured[-1]),real_delta_180_300=float(shift[end_300]),
            real_request_180_300=float(requested[:end_300+1].mean()),
            real_300_420_speed_max_min=np.quantile(measured[end_300:25],[1,0]).tolist(),
            real_330_420_request_max_min=np.quantile(requested[15:25],[1,0]).tolist(),
            sim_initial_speed=float(speeds[0,0]),sim_damping=float(sim.mj_model.dof_damping[right]),
            sim_friction=float(sim.mj_model.dof_frictionloss[right]),taper_parameters=taper_parameters,rows=rows,no_contacts=True,optimizer_updates=0)
        write_json(out/'report.json',report)
        np.savez_compressed(out/'wheel.npz',time_s=t,sim_speed=speeds,sim_torque=torque,real_speed=measured,real_torque_request=requested)
        print(json.dumps(report,indent=2))
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--check-speed-taper',action='store_true')
    args=p.parse_args()
    run(args.out,check_speed_taper=args.check_speed_taper)
