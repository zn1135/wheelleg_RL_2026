#!/usr/bin/env python3
"""Render full saved tracking-probe trajectories; no physics integration or learning."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from scripts.agent.stand.render_chuanliantui_stand import renderer_for


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.measurement/"result.json").read_text())
    rows = report["rows"]
    candidates = [r for r in rows if r["ready_at_8s"] and r["complete_tracking_window"]
                  and .09 <= abs(r["requested_vx_m_s"]) <= .31]
    if candidates:
        best = min(candidates, key=lambda r: (not r["final_drive_balance_1s"],
            abs(r["requested_vx_m_s"]) < .29, r["vx_mean_abs_error_m_s"]))
    else:
        best = rows[0]
    failed = [r for r in rows if not r["ready_at_8s"] or not r["final_drive_balance_1s"]]
    # 优先展示训练范围内的失败，不能把未训练的-0.5m/s外推失败当成本范围的失败。
    failed.sort(key=lambda r: not (.09 <= abs(r["requested_vx_m_s"]) <= .31))
    chosen = [best] + ([failed[0]] if failed and failed[0]["env"] != best["env"] else [])
    with np.load(args.measurement/"qpos.npz") as data:
        qpos, times = data["qpos"], data["time_s"]
    with np.load(args.measurement/"trace.npz") as data:
        values, active = data["values"], data["active"]
    model = mujoco.MjModel.from_binary_path(str(args.measurement/"scene.mjb"))
    renderer, frame = renderer_for(model, 800, 600)
    video = args.output/"model3999-tracking.mp4"
    frames = round(float(times[-1])*30)
    try:
        with imageio.get_writer(video, fps=30, codec="libx264", quality=8, macro_block_size=1) as writer:
            for k in range(frames):
                t = k/30.
                index = min(int(np.searchsorted(times, t)), len(times)-1)
                step = max(0, min(index-1, len(values)-1))
                canvas = Image.new("RGB", (800*len(chosen), 600))
                for column, row in enumerate(chosen):
                    env = row["env"]
                    end = row["termination_s"]
                    ended = end is not None and t >= end
                    case_index = min(index, int(np.searchsorted(times, end))) if ended else index
                    case_step = max(0, min(case_index-1, len(values)-1))
                    status = "READY" if row["ready_at_8s"] else "NOT READY AT8s"
                    if ended or not active[step, env]:
                        status = "FIRST EPISODE ENDED"
                    label = (f"model3999 | env{env} | {status} | t={t:.2f}s | "
                             f"cmd={values[case_step,env,0]:+.2f} vx={values[case_step,env,1]:+.2f}m/s")
                    canvas.paste(frame(qpos[case_index, env], label), (800*column, 0))
                writer.append_data(np.asarray(canvas))
                if k in (0, 300, 570):
                    canvas.save(args.output/f"frame-{k:03d}.png")
    finally:
        renderer.close()
    # Decode every frame before delivering the local video.
    reader = imageio.get_reader(video)
    decoded = 0
    try:
        for image in reader:
            assert image.shape[:2] == (600, 800*len(chosen))
            decoded += 1
    finally:
        reader.close()
    assert decoded == frames
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(video=str(video.resolve()), sha256=digest(video), frames=frames, decoded_frames=decoded,
        fps=30, seconds=float(times[-1]), env_ids=[r["env"] for r in chosen],
        checkpoint_sha256=report["checkpoint_sha256"], source_qpos_sha256=digest(args.measurement/"qpos.npz"),
        parameter_updates=0, source_sha256=digest(Path(__file__)),
        scope="Saved22s grouped-command probe; terminated cases freeze at first end, no velocity-pass threshold implied")
    (args.output/"video.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
