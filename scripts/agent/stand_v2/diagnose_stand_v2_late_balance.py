#!/usr/bin/env python3
"""Replay the original ten late-balance cases; record actual contact-point slip, no learning."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from scripts.agent.stand_v2 import evaluate_stand_v2_phase_mix as evaluation
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest


FIELDS = ("left_support_N", "right_support_N", "base_force_N", "roll_rad", "pitch_rad",
    "omega_x", "omega_y", "omega_xy", "height_m", "vx", "vy", "vz", "x", "y", "z",
    "gravity_z", "live", "stood", "slip_left", "slip_right", "friction_util_left", "friction_util_right",
    "contact_normal_left", "contact_normal_right", "contacts_left", "contacts_right",
    "wheel_dq_left", "wheel_dq_right", "wheel_tau_left", "wheel_tau_right",
    "push_fx", "push_fy", "push_fz", "push_count", "command_vx")

BASE_RECORDER = evaluation.Recorder


class DetailedRecorder(BASE_RECORDER):
    def __init__(self, raw, selected, output):
        super().__init__(raw)
        self.ids = torch.as_tensor(selected, device=raw.device)
        self.trace = torch.empty((20000, len(selected), len(FIELDS)), device=raw.device)
        self.qpos = [raw.sim.data.qpos[self.ids].clone()]
        self.world_to_case = torch.full((raw.num_envs,), -1, device=raw.device, dtype=torch.long)
        self.world_to_case[self.ids] = torch.arange(len(selected), device=raw.device)
        robot = raw.scene["robot"]
        local, _ = robot.find_bodies(("lfwheel", "rfwheel"), preserve_order=True)
        self.wheel_bodies = robot.indexing.body_ids[local]
        self.root_body = robot.indexing.root_body_id
        model = raw.sim.mj_model
        assert int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        self.geom_wheel = torch.full((model.ngeom,), -1, dtype=torch.long, device=raw.device)
        for wheel, body in enumerate(self.wheel_bodies.cpu().tolist()):
            self.geom_wheel[torch.tensor(model.geom_bodyid == body, device=raw.device)] = wheel
        self.is_floor = torch.tensor(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, device=raw.device)
        joint, _ = robot.find_joints(("lfwheel", "rfwheel"), preserve_order=True)
        self.wheel_v = robot.indexing.joint_v_adr[joint]
        self.fixtures = []
        mujoco.mj_saveModel(model, str(output/"scene.mjb"), None)

    def contacts(self):
        raw, ids = self.raw, self.ids
        d, c = raw.sim.data, raw.sim.data.contact
        n = int(d.nacon[0])
        worlds, geoms = c.worldid[:n].long(), c.geom[:n].long()
        case = self.world_to_case[worlds]
        side0, side1 = self.geom_wheel[geoms[:, 0]], self.geom_wheel[geoms[:, 1]]
        use0 = (side0 >= 0) & self.is_floor[geoms[:, 1]]
        use1 = (side1 >= 0) & self.is_floor[geoms[:, 0]]
        addr = c.efc_address[:n, :3].long()
        use = (case >= 0) & (use0 | use1) & (c.dim[:n] >= 3) & (addr >= 0).all(-1)
        contact_ids = use.nonzero(as_tuple=False).flatten()
        case, worlds, addr = case[use], worlds[use], addr[use]
        side = torch.where(use0, side0, side1)[use]
        point = c.pos[:n][use]
        origin = d.xpos[ids[:, None], self.wheel_bodies]
        cvel = d.cvel[ids[:, None], self.wheel_bodies]
        center = d.subtree_com[ids, self.root_body][:, None]
        omega = cvel[..., :3]
        velocity = cvel[..., 3:] - torch.cross(omega, center-origin, dim=-1)
        # 平地静止：轮面实际接触点速度 = 轮体原点速度 + 角速度叉乘接触偏移。
        relative = velocity[case, side] + torch.cross(omega[case, side], point-origin[case, side], dim=-1)
        frame = c.frame[:n][use]
        local_velocity = torch.einsum("nij,nj->ni", frame, relative)
        slip = local_velocity[:, 1:].norm(dim=-1)
        force = d.efc.force[worlds[:, None], addr]
        normal = force[:, 0].clamp_min(0.)
        mu = c.friction[:n][use, :2]
        utilization = (force[:, 1:] / mu.clamp_min(1e-8)).norm(dim=-1)/normal.clamp_min(1e-8)
        flat = case*2+side
        sums = torch.zeros((4, len(ids)*2), device=raw.device)
        sums[0].scatter_add_(0, flat, normal)
        sums[1].scatter_add_(0, flat, normal*slip)
        sums[2].scatter_add_(0, flat, normal*utilization)
        sums[3].scatter_add_(0, flat, (normal > 1.).float())
        slip_mean = sums[1]/sums[0].clamp_min(1e-8)
        util_mean = sums[2]/sums[0].clamp_min(1e-8)
        if self.count >= 17500 and self.count % 250 == 0:
            self.fixtures.append(dict(point=point.cpu().numpy(), origin=origin[case,side].cpu().numpy(),
                velocity=velocity[case,side].cpu().numpy(), omega=omega[case,side].cpu().numpy(),
                frame=frame.cpu().numpy(), slip=slip.cpu().numpy(), force=force.cpu().numpy(),
                mu=mu.cpu().numpy(), utilization=utilization.cpu().numpy(),
                contact_ids=contact_ids.cpu().numpy()))
        return [v.reshape(len(ids),2) for v in (slip_mean, util_mean, sums[0], sums[3])]

    def sample(self):
        super().sample()
        raw, ids, d = self.raw, self.ids, self.raw.sim.data
        q, vel = d.qpos[ids][:,self.root_q], d.qvel[ids][:,self.root_v]
        w,x,y,z = q[:,3:].unbind(-1)
        gx,gy,gz = 2*(w*y-x*z),-2*(y*z+w*x),-1+2*(x*x+y*y)
        roll, pitch = torch.atan2(-gy,-gz), gx.clamp(-1.,1.).asin()
        support = -raw.scene["wheel_contact"].data.force[ids,:,2]
        base = raw.scene["base_contact"].data.force[ids].norm(dim=-1).amax(-1)
        slip, util, normal, count = self.contacts()
        state = raw.command_manager.get_term("standup")
        physical = raw._standup_randomization
        self.trace[self.count-1] = torch.cat((support, base[:,None], roll[:,None], pitch[:,None],
            vel[:,3:5], vel[:,3:5].norm(dim=-1,keepdim=True),
            (q[:,2]-raw.scene.env_origins[ids,2])[:,None], vel[:,:3], q[:,:3],
            gz[:,None], self.live[ids,None].float(), state.has_stood[ids,None].float(),
            slip, util, normal, count, d.qvel[ids[:,None],self.wheel_v],
            d.qfrc_actuator[ids[:,None],self.wheel_v], physical.push_force[ids],
            physical.push_count[ids,None].float(), state.command[ids,0,None]),dim=-1)
        if self.count % 5 == 0:
            self.qpos.append(d.qpos[ids].clone())


def analyze(output, original, selected):
    with np.load(output/"detail.npz") as saved:
        trace, weight = saved["trace"],saved["weight"]
    with np.load(output/"episodes.npz") as episodes:
        reproduced = episodes["ever_stood"][selected] & ~episodes["passed"][selected]
    rows = []
    for i, env in enumerate(selected):
        v = trace[:,i]; tail = v[-2500:]; final = v[-500:]
        conditions = dict(each_wheel=(v[:,:2] >= max(.1*weight[i],5.)).all(-1),
            total_support=((v[:,:2].sum(-1) >= .8*weight[i]) & (v[:,:2].sum(-1) <= 1.2*weight[i])),
            base_clear=v[:,2] <= 1., tilt=np.abs(v[:,3:5]).max(-1) <= np.deg2rad(35.),
            angular_speed=v[:,7] <= .5, finite=np.isfinite(v).all(-1), live=v[:,16]>.5)
        balanced = np.logical_and.reduce(list(conditions.values()))
        last_bad = np.flatnonzero(~balanced)
        hold = len(v) if not len(last_bad) else len(v)-1-int(last_bad[-1])
        pushing = np.linalg.norm(v[:,30:33],axis=-1)>1e-4
        starts=np.flatnonzero(pushing & ~np.r_[False,pushing[:-1]])
        loaded=tail[:,22:24]>5.
        slip=tail[:,18:20][loaded]; util=tail[:,20:22][loaded]
        with np.load(output/"episodes.npz") as episodes:
            assert hold == int(episodes["final_hold_steps"][env])
        rows.append(dict(env=int(env),original_late_failure=True,reproduced_late_failure=bool(reproduced[i]),
            final_hold_s=hold*.002, final1_condition_bad_samples={k:int((~m[-500:]).sum()) for k,m in conditions.items()},
            last5_condition_bad_fraction={k:float((~m[-2500:]).mean()) for k,m in conditions.items()},
            last5_speed_mean=float(np.linalg.norm(tail[:,9:11],axis=-1).mean()),
            last5_xy_excursion=float(np.linalg.norm(tail[:,12:14]-tail[0,12:14],axis=-1).max()),
            last5_max_tilt_deg=float(np.rad2deg(np.abs(tail[:,3:5]).max())),
            last5_max_omega_x=float(np.abs(tail[:,5]).max()),last5_max_omega_y=float(np.abs(tail[:,6]).max()),
            last5_base_max_N=float(tail[:,2].max()),
            loaded_slip_median_m_s=float(np.median(slip)) if len(slip) else None,
            loaded_slip_p95_m_s=float(np.quantile(slip,.95)) if len(slip) else None,
            loaded_slip_above005_fraction=float((slip>.05).mean()) if len(slip) else None,
            loaded_friction_util_p95=float(np.quantile(util,.95)) if len(util) else None,
            last_push_start_s=float((starts[-1]+1)*.002) if len(starts) else None,
            push_active_in_final1=bool(pushing[-500:].any()),
            last_condition_break_s=float((last_bad[-1]+1)*.002) if len(last_bad) else None))
    result=dict(original_case_ids=selected.tolist(),reproduced_late_failures=int(reproduced.sum()),
        rows=rows,parameter_updates=0,scope="Same-initial-state rerun, not the original saved trajectory; contact slip is force-weighted tangential point velocity",
        original_episodes_sha256=digest(original/"episodes.npz"),detail_sha256=digest(output/"detail.npz"),source_sha256=digest(__file__))
    (output/"late-analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def main(args):
    with np.load(args.original/"episodes.npz") as saved:
        selected=np.flatnonzero(saved["ever_stood"] & ~saved["passed"])
    assert len(selected)==10
    recorders=[]
    def factory(raw):
        recorder=DetailedRecorder(raw,selected,args.output)
        recorders.append(recorder)
        return recorder
    # 复用原评估入口，不改变初态、随机化、动作、终止或自动复位处理。
    with patch.object(evaluation,"Recorder",factory):
        evaluation.main(argparse.Namespace(checkpoint=args.checkpoint,output=args.output,
            conditions="training",seed=42,num_envs=128,device=args.device))
    rec=recorders[0]
    assert rec.count==20000
    with np.load(args.original/"initial.npz") as a,np.load(args.output/"initial.npz") as b:
        assert set(a.files)==set(b.files)
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    np.savez_compressed(args.output/"detail.npz",trace=rec.trace.cpu().numpy(),fields=FIELDS,
        weight=rec.weight[rec.ids].cpu().numpy(),env_ids=selected)
    np.savez_compressed(args.output/"qpos.npz",qpos=torch.stack(rec.qpos).cpu().numpy(),
        time_s=np.arange(4001)*.01,env_ids=selected)
    fixtures={k:np.concatenate([f[k] for f in rec.fixtures]) for k in rec.fixtures[0]}
    reconstructed=fixtures["velocity"]+np.cross(fixtures["omega"],fixtures["point"]-fixtures["origin"])
    slip=np.linalg.norm(np.einsum("nij,nj->ni",fixtures["frame"],reconstructed)[:,1:],axis=-1)
    assert np.allclose(slip,fixtures["slip"],atol=1e-6)
    utilization=np.linalg.norm(fixtures["force"][:,1:]/np.maximum(fixtures["mu"],1e-8),axis=-1)/np.maximum(fixtures["force"][:,0],1e-8)
    assert np.allclose(utilization,fixtures["utilization"],atol=1e-4)
    np.savez_compressed(args.output/"contact-fixtures.npz",**fixtures)
    analyze(args.output,args.original,selected)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    main(parser.parse_args())
