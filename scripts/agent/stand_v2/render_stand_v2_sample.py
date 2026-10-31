#!/usr/bin/env python3
"""Render a recorded stand_v2 sample at its original simulation speed."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from scripts.agent.stand.render_chuanliantui_stand import renderer_for


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.recording/"recording.json").read_text())
    report = json.loads((args.recording/"result.json").read_text())
    with np.load(args.recording/"sample-qpos.npz") as saved:
        qpos, times = saved["qpos"], saved["time_s"]
    with np.load(args.recording/"measurements.npz") as saved:
        speed = saved["values"][:, metadata["env"], 5]
    model = mujoco.MjModel.from_binary_path(str(args.recording/"scene.mjb"))
    assert len(qpos) == len(times) and qpos.shape[1] == model.nq
    fps = 30
    frame_times = np.arange(round(metadata["seconds"]*fps))/fps
    renderer, frame = renderer_for(model, 800, 600)
    model_name = Path(report["checkpoint"]).stem.replace("model_", "model")
    output = args.output/f'env{metadata["env"]}-{model_name}.mp4'
    try:
        with imageio.get_writer(output, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1) as writer:
            for k, time_s in enumerate(frame_times):
                index = min(int(np.searchsorted(times, time_s)), len(qpos)-1)
                v = speed[max(0, min(index-1, len(speed)-1))]
                label = f'Env {metadata["env"]} | {model_name} | t={time_s:.2f}s | planar speed={v:.2f} m/s'
                canvas = frame(qpos[index], label)
                writer.append_data(np.asarray(canvas))
                if k == 0:
                    canvas.save(args.output/"preview.png")
    finally:
        renderer.close()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = dict(video=str(output.resolve()), video_sha256=sha(output), frames=len(frame_times),
        fps=fps, duration_s=metadata["seconds"], env=metadata["env"],
        source_qpos_sha256=sha(args.recording/"sample-qpos.npz"),
        checkpoint_sha256=report["checkpoint_sha256"], parameter_updates=0,
        scope="Render of same-initial-state diagnostic rerun, not original frame-for-frame capture")
    (args.output/"video.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
