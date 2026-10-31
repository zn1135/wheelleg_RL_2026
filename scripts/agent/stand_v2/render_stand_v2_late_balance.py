#!/usr/bin/env python3
"""Plot the ten rerun tails and render saved trajectories; no integration or learning."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","egl")
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest
from scripts.agent.stand.render_chuanliantui_stand import renderer_for


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    result=json.loads((args.recording/"late-analysis.json").read_text())
    with np.load(args.recording/"detail.npz") as data:
        trace,weight,ids=data["trace"],data["weight"],data["env_ids"]
    time=(np.arange(20000)+1)*.002
    start=17500
    for i,row in enumerate(result["rows"]):
        v=trace[:,i]; t=time[start:]; tail=v[start:]
        fig,axes=plt.subplots(5,1,figsize=(11,13),sharex=True)
        axes[0].plot(t,np.rad2deg(tail[:,3]),label="roll")
        axes[0].plot(t,np.rad2deg(tail[:,4]),label="pitch")
        axes[0].set_ylabel("Angle (deg)")
        axes[0].set_title(f"Actual tilt; acceptance limit35deg, tail max={row['last5_max_tilt_deg']:.2f}deg")
        axes[1].plot(t,np.abs(tail[:,5]),label="abs omega_x")
        axes[1].plot(t,np.abs(tail[:,6]),label="abs omega_y")
        axes[1].plot(t,tail[:,7],label="omega_xy norm",color="black",alpha=.6)
        axes[1].axhline(.5,color="red",ls="--",label="0.5 gate")
        axes[1].set_ylabel("Angular speed (rad/s)")
        axes[2].plot(t,tail[:,0]/weight[i],label="left / weight")
        axes[2].plot(t,tail[:,1]/weight[i],label="right / weight")
        axes[2].plot(t,tail[:,:2].sum(-1)/weight[i],label="total / weight")
        for value in (.1,.8,1.2):axes[2].axhline(value,color="grey",ls="--",alpha=.5)
        axes[2].set_ylabel("Wheel support ratio")
        axes[3].plot(t,tail[:,18],label="left contact slip")
        axes[3].plot(t,tail[:,19],label="right contact slip")
        axes[3].axhline(.05,color="grey",ls="--",label="0.05 diagnostic reference")
        axes[3].set_ylabel("Contact slip (m/s)")
        axes[3].set_title(f"Loaded-contact friction utilization p95={row['loaded_friction_util_p95']:.3f}")
        axes[4].plot(t,tail[:,9],label="vx world")
        axes[4].plot(t,tail[:,10],label="vy world")
        axes[4].plot(t,np.linalg.norm(tail[:,9:11],axis=-1),label="planar speed")
        axes[4].set_ylabel("Base speed (m/s)")
        axes[4].set_xlabel("Simulation time (s); shaded grey=final1s, orange=external push")
        push=np.linalg.norm(v[:,30:33],axis=-1)>1e-4
        begins=np.flatnonzero(push & ~np.r_[False,push[:-1]])
        ends=np.flatnonzero(push & ~np.r_[push[1:],False])
        for axis in axes:
            axis.axvspan(39,40,color="grey",alpha=.08)
            for a,b in zip(begins,ends):
                if time[b]>=35:axis.axvspan(time[a],time[b],color="orange",alpha=.2)
            axis.grid(alpha=.2);axis.legend(loc="upper right",fontsize=8)
            axis.set_xlim(35,40)
        outcome="late failure reproduced" if row["reproduced_late_failure"] else "late failure NOT reproduced"
        fig.suptitle(f"env{row['env']} | same-initial-state rerun | {outcome} | final hold={row['final_hold_s']:.3f}s\n"
            +str(row["final1_condition_bad_samples"]),fontsize=10)
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(args.output/f"env{row['env']:03d}-tail.png",dpi=120)
        plt.close(fig)
    reproduced=[r for r in result["rows"] if r["reproduced_late_failure"]]
    primary=max(reproduced or result["rows"],key=lambda r:r["final1_condition_bad_samples"]["angular_speed"])
    changed=[r for r in result["rows"] if not r["reproduced_late_failure"] and r["env"]!=primary["env"]]
    alternatives=changed or [r for r in result["rows"] if r["env"]!=primary["env"]]
    chosen=[primary]+alternatives[:1]
    with np.load(args.recording/"qpos.npz") as saved:qpos,times=saved["qpos"],saved["time_s"]
    with np.load(args.recording/"episodes.npz") as saved:ended=saved["first_end_s"]
    model=mujoco.MjModel.from_binary_path(str(args.recording/"scene.mjb"))
    renderer,frame=renderer_for(model,800,600)
    video=args.output/"late-balance-rerun.mp4"
    try:
        with imageio.get_writer(video,fps=30,codec="libx264",quality=8,macro_block_size=1) as writer:
            for k in range(1200):
                t=k/30.;index=min(int(np.searchsorted(times,t)),len(times)-1)
                canvas=Image.new("RGB",(800*len(chosen),600))
                for col,row in enumerate(chosen):
                    env=row["env"];local=int(np.flatnonzero(ids==env)[0]); case=index
                    if np.isfinite(ended[env]) and t>=ended[env]:case=min(case,int(np.searchsorted(times,ended[env])))
                    tick=min(round(times[case]/.002),19999)
                    omega=trace[tick,local,7]
                    status="LATE AGAIN" if row["reproduced_late_failure"] else "PASSED RERUN"
                    label=f"3999 rerun | env{env} | {status} | t={t:.2f}s | omega_xy={omega:.2f}rad/s"
                    canvas.paste(frame(qpos[case,local],label),(col*800,0))
                writer.append_data(np.asarray(canvas))
                if k in (0,1050,1170):canvas.save(args.output/f"frame-{k:04d}.png")
    finally:renderer.close()
    reader=imageio.get_reader(video)
    try:
        decoded=sum(1 for _ in reader)
    finally:reader.close()
    assert decoded==1200
    report=dict(video=str(video.resolve()),sha256=digest(video),fps=30,frames=1200,decoded_frames=decoded,
        seconds=40,selected_envs=[r["env"] for r in chosen],parameter_updates=0,
        original_vs_rerun="Original only had episode statistics; all plots/video are explicitly a same-initial-state rerun",
        detail_sha256=digest(args.recording/"detail.npz"),source_sha256=digest(__file__))
    (args.output/"render.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
