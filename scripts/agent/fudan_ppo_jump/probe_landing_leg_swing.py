"""实机第二跳落地的腿摆角、腿长和机身姿态反事实对照。"""
import argparse
import csv
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ct_mjlab.envs.chuanliantui.model import closed_chain_residual
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import jump_config
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.reset_geometry import mesh_vertices
from .compare_landing_lead import ALIGN, ARCHIVE
from .probe_takeoff_work import ROOT, sha, write_json
from .render_real_log import MirrorPose, MODEL, H7_PROJECT_CONFIGS

REFERENCE = ROOT/'artifacts/fudan_ppo_jump/20261125-flight-wheel-02'
REPLACEMENTS = tuple(product((False, True), repeat=3))
STEPS = 12


class LandingGeometry:
    def __init__(self):
        self.solver = MirrorPose(MODEL,H7_PROJECT_CONFIGS['H7_ros'])
        self.model, self.data = self.solver.model, self.solver.data
        self.wheels = [self.model.body(n).id for n in ('lfwheel','rfwheel')]
        self.hips = [self.model.joint(n).id for n in ('lf0','rf0')]
        self.base = self.model.body('base_link').id
        self.wheel_geoms = [next(g for g in range(self.model.ngeom) if self.model.geom_bodyid[g]==bid and self.model.geom_contype[g]) for bid in self.wheels]
        self.active = [self.model.joint(n).dofadr[0] for n in ('lf0','lf00','rf0','rf00','lfwheel','rfwheel')]
        self.passive = np.setdiff1d(np.arange(6,self.model.nv),self.active)
        self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

    def describe(self, q):
        d = self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model,d)
        frame = d.xmat[self.base].reshape(3,3)
        leg = (d.xpos[self.wheels]-d.xanchor[self.hips]) @ frame
        meshes = [mesh_vertices(self.model,d,g) for g in self.wheel_geoms]
        lowest = np.array([verts[np.argmin(verts[:,2])] for verts in meshes])
        mujoco.mj_comPos(self.model,d)
        return dict(swing=np.arctan2(leg[:,0],-leg[:,2]), length=np.linalg.norm(leg[:,[0,2]],axis=1),
                    bottom=lowest[:,2], rpy=Rotation.from_matrix(frame).as_euler('xyz'), fore_aft=leg[:,0],
                    contact_in_body_x=((lowest-d.subtree_com[self.base]) @ frame)[:,0])

    def mix(self, q_sim, q_real, use_swing, use_length, use_body):
        sim, real = self.describe(q_sim), self.describe(q_real)
        source = q_real if use_length else q_sim
        source_swing = real['swing'] if use_length else sim['swing']
        goal_swing = real['swing'] if use_swing else sim['swing']
        self.data.qpos[:] = source
        # 前后主动轴有共同的旋转轴线。两轴同量转动可改变摆角并保持腿长。
        for side, sign in enumerate((-1,1)):
            ids = self.solver.drive_q[side]
            assert self.solver._solve_leg(side,source[ids]+sign*(goal_swing[side]-source_swing[side]))
        q = self.data.qpos.copy()
        q[:7] = q_sim[:7]
        q[self.solver.wheel_q] = q_sim[self.solver.wheel_q]
        if use_body:
            q[3:7] = Rotation.from_euler('xyz',[*real['rpy'][:2],sim['rpy'][2]]).as_quat()[[3,0,1,2]]
        result = self.describe(q)
        np.testing.assert_allclose(result['swing'],goal_swing,atol=.001,rtol=0)
        np.testing.assert_allclose(result['length'],real['length'] if use_length else sim['length'],atol=.001,rtol=0)
        assert closed_chain_residual(self.model,self.data)<1e-6
        # 将各例较低轮底设为同一初始净空，两轮高度差完整保留。
        q[2] += sim['bottom'].min()-result['bottom'].min()
        return q,result

    def velocity(self,q,v):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model,self.data)
        jacobian = self.data.efc_J.reshape(-1,self.model.nv)[self.data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        v = v.copy()
        v[self.passive] = 0
        v[self.passive] = np.linalg.lstsq(jacobian[:,self.passive], -jacobian@v, rcond=None)[0]
        assert abs(jacobian@v).max()<1e-7
        return v


def readable(result):
    return dict(swing_deg=np.degrees(result['swing']).tolist(), fore_aft_cm=(result['fore_aft']*100).tolist(),
                leg_length_cm=(result['length']*100).tolist(), roll_pitch_deg=np.degrees(result['rpy'][:2]).tolist(),
                bottom_position_relative_to_com_forward_cm=(result['contact_in_body_x']*100).tolist(),
                left_minus_right_bottom_cm=float((result['bottom'][0]-result['bottom'][1])*100))


@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False)
    cap = json.loads((REFERENCE/'capture.json').read_text())
    alignment = json.loads(ALIGN.read_text())
    trial = next(t for t in alignment['trials'] if t['trial']==2)
    archive_hash = sha(ARCHIVE)
    assert archive_hash==alignment['archive_sha256']==cap['archive_sha']
    reconstruction = ARCHIVE.parent/'displayed-poses.npz'
    assert json.loads((reconstruction.parent/'report.json').read_text())['archive_sha256']==archive_hash
    with np.load(reconstruction) as measured:
        select = measured['case']==1
        real_poses = measured['qpos'][select]
        real_time = (measured['board_us'][select]-trial['first_jump_board_us'])/1000
    real_contact = 1000*(np.asarray(trial['touchdown_bracket_frames'])/alignment['fps']-trial['video_offset_s'])
    report = json.loads((REFERENCE/'report.json').read_text())
    geometry = LandingGeometry()
    frames = []
    with np.load(REFERENCE/'gpu.npz') as raw:
        for reference_arm in ('walk','right_wheel_log'):
            j = next(i for i,c in enumerate(cap['cases']) if c['scenario']=='imu' and c['arm']==reference_arm)
            touch = report['nominal'][reference_arm]['first_wheel_touchdown_ms']
            tick = (touch-15)//10
            qs,vs = raw['q'][tick,j],raw['v'][tick,j]
            for contact_ms in (*real_contact,real_contact.mean()):
                idx = int(np.argmin(abs(real_time-(contact_ms+tick*10-touch))))
                qr = real_poses[idx]
                for swing,length,body in REPLACEMENTS:
                    q,details = geometry.mix(qs,qr,swing,length,body)
                    v = geometry.velocity(q,vs)
                    label = '+'.join(key for key,applied in zip(('swing','length','body'),(swing,length,body)) if applied) or 'baseline'
                    row = dict(reference=reference_arm,arm=label,sim_start_ms=tick*10,real_pose_ms=float(real_time[idx]),
                               real_contact_assumption_ms=float(contact_ms),**readable(details))
                    frames.append((row,q,v,raw['action'][tick:tick+STEPS,j].copy()))
    env = JumpEnv(len(frames),823,'cuda:0',False,cfg=jump_config(),record_flat=True)
    device = env.device
    def tensor(x):
        return torch.as_tensor(x,dtype=torch.float32,device=device)
    try:
        env.reset_idx(torch.arange(len(frames),device=device))
        sim = env.env.sim
        assert sim.mj_model.nq==geometry.model.nq and sim.mj_model.nv==geometry.model.nv
        for j in range(1,geometry.model.njnt):
            model_joint=sim.mj_model.joint('robot/'+geometry.model.joint(j).name)
            assert model_joint.qposadr[0]==geometry.model.jnt_qposadr[j] and model_joint.dofadr[0]==geometry.model.jnt_dofadr[j]
        q0 = np.stack([q for _,q,_,_ in frames])
        v0 = np.stack([v for _,_,v,_ in frames])
        sim.data.qpos[:] = tensor(q0)
        sim.data.qvel[:] = tensor(v0)
        sim.data.ctrl.zero_()
        sim.data.qacc_warmstart.zero_()
        sim.forward()
        env.env.scene.update(dt=0.)
        sim.sense()
        env.commands[:] = tensor([0.,0.,.22])
        actions = tensor(np.stack([a for _,_,_,a in frames],axis=1))
        pitch, traces = [], []
        for action in actions:
            env.substeps.clear()
            env.step(action,auto_reset=False,resample=False)
            assert len(env.substeps)==10 and not sim.wp_data.overflow.numpy().any()
            traces.append(torch.stack(env.substeps).cpu().numpy())
            pitch.append(Rotation.from_quat(sim.data.qpos[:,[4,5,6,3]].cpu().numpy()).as_euler('xyz',degrees=True)[:,1])
        trace, pitch = np.concatenate(traces), np.asarray(pitch)
        assert np.isfinite(trace).all() and np.isfinite(pitch).all()
        rows = []
        for j,(row,_,_,_) in enumerate(frames):
            wheel_hits=(trace[:,j,8:10]>.5)
            hit=np.flatnonzero(wheel_hits.any(1))
            first = int(hit[0]) if len(hit) else None
            who = wheel_hits[first] if first is not None else (False,False)
            row.update(first_touch_after_init_ms=first,first_wheel='both' if all(who) else ('left' if who[0] else ('right' if who[1] else None)),
                       nonwheel_ms=int((trace[:,j,10]>1).sum()),pitch_final_deg=float(pitch[-1,j]))
            rows.append(row)
        write_json(out/'report.json',dict(cases=rows, archive_sha=archive_hash, pose_sha=sha(reconstruction),
                   reference=str(REFERENCE.relative_to(ROOT)),real_baseline=readable(geometry.describe(real_poses[np.argmin(abs(real_time-real_contact.mean()))])),
                   real_contact_ms=real_contact.tolist(),same_tape_per_reference=True,physics_ms=1,pd_ms=5,action_ms=10,optimizer_updates=0))
        with (out/'cases.csv').open('x') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        np.savez_compressed(out/'traces.npz',trace=trace,pitch=pitch,q0=q0,v0=v0,actions=actions.cpu().numpy())
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    run(p.parse_args().out)
