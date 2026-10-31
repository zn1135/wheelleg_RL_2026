#!/usr/bin/env python3
"""Render a passed and failed case from one saved acceptance; never resimulate."""

import argparse
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
    report = json.loads((args.acceptance/"result.json").read_text())
    passed = [r for r in report["rows"] if r["passed"]]
    failed = [r for r in report["rows"] if not r["passed"]]
    # Include a body-supported failure when one exists, rather than only inversion.
    body = [r for r in failed if r["last5_median_base_force_N"] > 1. and not r["ever_inverted"]]
    rows = ([passed[0]] if passed else []) + ([body[0] if body else failed[0]] if failed else [])
    with np.load(args.acceptance/"qpos.npz") as saved:
        qpos, times = saved["qpos"], saved["time_s"]
    model = mujoco.MjModel.from_binary_path(str(args.acceptance/"scene.mjb"))
    renderer, frame = renderer_for(model, 800, 600)
    name = Path(report["checkpoint"]).stem.replace("model_", "model")
    video = args.output/f"{name}-pass-fail.mp4"
    try:
        with imageio.get_writer(video, fps=30, codec="libx264", quality=8, macro_block_size=1) as writer:
            for k in range(600):
                t = k/30.
                index = min(int(np.searchsorted(times, t)), len(times)-1)
                sheet = Image.new("RGB", (800*len(rows), 600))
                for column, row in enumerate(rows):
                    status = "PASS" if row["passed"] else "FAIL"
                    label = f'{name} | seed={report["seed"]} | env={row["env"]} | {status} | t={t:.2f}s'
                    sheet.paste(frame(qpos[index, row["env"]], label), (column*800, 0))
                writer.append_data(np.asarray(sheet))
                if k in (0, 60, 300, 570):
                    sheet.save(args.output/f"frame-{k:03d}.png")
    finally:
        renderer.close()
    from hashlib import sha256
    (args.output/"video.json").write_text(json.dumps(dict(video=str(video.resolve()),
        sha256=sha256(video.read_bytes()).hexdigest(), frames=600, fps=30, duration_s=20.,
        env_ids=[r["env"] for r in rows], checkpoint_sha256=report["checkpoint_sha256"],
        source_qpos_sha256=report["qpos_sha256"],
        scope="Exact saved acceptance trajectory, full 20s at original speed; no rerun"), indent=2)+"\n")
    print(video.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
