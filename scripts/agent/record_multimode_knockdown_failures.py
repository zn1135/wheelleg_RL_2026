#!/usr/bin/env python3
"""记录击倒评估；也可按修订的交付定义直接渲染已保存的本次运行轨迹。

评估直接调用 assess.run_seed；step 包装只读取状态，不改变构建、复位和随机流。
渲染阶段只做 CPU mj_forward，不重新积分物理，不调用策略。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

import assess_multimode_knockdown_recovery as assess
from mjlab.viewer.model_sync import (
    VIEWER_MODEL_FIELDS, disable_model_sameframe_shortcuts,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "logs/mjlab/evaluation/kd-magcurve-rr-4850/result.json"
OUTPUT = ROOT / "logs/mjlab/evaluation/kd-failure-videos-4850"
CHECKPOINT_SHA = "de492a8ac361d9c6ef14b4531a9bb14a79d1af13b3ffe11cb9b09ac5968b2cc5"
TARGETS = (195, 198, 311, 318, 319, 93, 58)
TERRAINS = ("flat", "slope", "step", "stair")
FIELDS = ("recovered", "last5_pure_v2_fraction", "final_tilt_deg", "max_tilt_deg")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def record(args, output, reference):
    buffers = {name: [] for name in (
        "qpos", "qvel", "mocap_pos", "mocap_quat", "gravity", "linear",
        "angular", "base_force", "wheel_force", "all_force", "height", "fallen",
    )}
    original_build = assess.build_env
    env_args = SimpleNamespace(
        checkpoint=Path(reference["checkpoint"]), envs_per_combo=32, recovery_window=12.,
        recover_deadline=12., episode_seconds=20., device=args.device,
    )

    def build(*build_args):
        built = original_build(*build_args)
        env = built[0]
        if env.num_envs != 384:
            raise ValueError(f"环境数错误：{env.num_envs}")
        original_step = env.step
        step_count = 0
        fields = {}

        def step(actions):
            nonlocal step_count
            result = original_step(actions)
            # 在 assess 读取终止状态及执行 reset 之前记录；不向仿真写入任何值。
            state = env.command_manager.get_term("standup")
            robot = env.scene["robot"]
            if step_count == 0:
                rng_before = torch.cuda.get_rng_state(args.device)
                mujoco.mj_saveModel(env.sim.mj_model, str(output / "scene.mjb"))
                for name in sorted(env.sim.expanded_fields & VIEWER_MODEL_FIELDS):
                    fields["model_" + name] = getattr(env.sim.model, name)[list(TARGETS)].cpu().numpy().copy()
            sources = {
                "qpos": env.sim.data.qpos, "qvel": env.sim.data.qvel,
                "mocap_pos": env.sim.data.mocap_pos, "mocap_quat": env.sim.data.mocap_quat,
                "gravity": robot.data.projected_gravity_b,
                "linear": robot.data.root_link_lin_vel_b, "angular": robot.data.root_link_ang_vel_b,
                "base_force": env.scene["base_contact"].data.force,
                "wheel_force": env.scene["wheel_contact"].data.force,
                "all_force": env.scene["all_robot_contact"].data.force,
                "height": state.height_above_ground, "fallen": state.fallen,
            }
            for name, tensor in sources.items():
                buffers[name].append(tensor[list(TARGETS)].cpu().numpy().copy())
            if step_count == 0 and not torch.equal(rng_before, torch.cuda.get_rng_state(args.device)):
                raise RuntimeError("记录状态消耗了 torch CUDA RNG")
            step_count += 1
            if step_count % 100 == 0:
                print(f"RECORD {step_count}/1000", flush=True)
            return result

        env.step = step
        record.fields = fields
        return built

    assess.build_env = build
    assess.RECOVER_DEADLINE_S = 12.
    assess.configure_torch_backends()
    try:
        rows, extras = assess.run_seed(env_args, 43, [12., 16., 20., 25., 30., 40.], [3., 9.])
    finally:
        assess.build_env = original_build
    np.savez_compressed(output / "recorded-states.npz", env_ids=np.array(TARGETS),
                        **{k: np.stack(v) for k, v in buffers.items()}, **record.fields)
    dump(output / "reproduced-episodes.json", {"episodes": rows, "extras": extras})
    return rows, extras


def verify(output, reference, rows, weight):
    expected = {r["env"]: r for r in reference["episodes"]}
    actual = {r["env"]: r for r in rows}
    comparisons, errors = [], []
    with np.load(output / "recorded-states.npz") as states:
        for col, env_id in enumerate(TARGETS):
            ref, row = expected[env_id], actual[env_id]
            # 独立用记录的传感器重算纯 v2 占比和倾角，不使用渲染结果作证据。
            count = round(row["duration_s"] / .02)
            gravity = states["gravity"][:count, col]
            tilt = np.maximum(np.abs(np.arctan2(-gravity[:, 1], -gravity[:, 2])),
                              np.abs(np.arcsin(np.clip(gravity[:, 0], -1, 1))))
            base = np.linalg.norm(states["base_force"][:count, col], axis=-1).max(-1)
            wheels = -states["wheel_force"][:count, col, :, 2]
            nonwheel = np.maximum(-states["all_force"][:count, col, :, 2].sum(-1) - wheels.sum(-1), 0.)
            v2 = ((tilt <= assess.TILT_MAX_RAD) & (base <= assess.BASE_FORCE_MAX_N)
                  & (wheels.min(-1) >= max(.1 * weight, 5.))
                  & (wheels.sum(-1) >= .8 * weight) & (wheels.sum(-1) <= 1.2 * weight)
                  & (nonwheel <= .05 * weight)
                  & (np.linalg.norm(states["linear"][:count, col], axis=-1) <= .15)
                  & (np.linalg.norm(states["angular"][:count, col], axis=-1) <= .5))
            duration = np.cumsum(np.full(count, .02, dtype=np.float32))
            tail = duration > 15.
            after_hit = duration > row["hit_time_s"]
            independent = {
                "last5_pure_v2_fraction": float(v2[tail].mean()) if tail.any() else None,
                "final_tilt_deg": float(np.degrees(tilt[-1])),
                "max_tilt_deg": float(np.degrees(tilt[after_hit].max())),
            }
            differences = {}
            for field in FIELDS:
                a, b = row[field], ref[field]
                if (a != b if isinstance(b, bool) or b is None else abs(a - b) > 1e-3):
                    differences[field] = {"reference": b, "reproduced": a}
            for field, value in independent.items():
                if value is None or abs(value - row[field]) > 1e-3:
                    differences["independent_" + field] = {"assess": row[field], "recorded_state": value}
            for field in ("seed", "magnitude", "knockdown_at_s", "terrain_kind", "termination"):
                if row[field] != ref[field]:
                    differences[field] = {"reference": ref[field], "reproduced": row[field]}
            comparisons.append({"env": env_id, "passed": not differences,
                                "reference": {f: ref[f] for f in FIELDS},
                                "reproduced": {f: row[f] for f in FIELDS},
                                "independent": independent, "differences": differences})
            if differences:
                errors.append({"env": env_id, **differences})
    report = {"passed": not errors, "tolerance": 1e-3, "comparisons": comparisons,
              "source_result_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "assess_script_sha256": hashlib.sha256(Path(assess.__file__).read_bytes()).hexdigest()}
    dump(output / "verification.json", report)
    if errors:
        print(json.dumps(errors, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit("复现失败：停止，不输出视频。差异见 verification.json")
    print("VERIFIED all 7 samples within 1e-3", flush=True)
    return actual


def render(output, rows, ffprobe):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 1.6, -18., 90.
    sheet = Image.new("RGB", (1280, 7 * 392), "#eeeeee")
    videos = []
    with np.load(output / "recorded-states.npz") as archive:
        states = {key: archive[key] for key in archive.files}
        model = mujoco.MjModel.from_binary_path(str(output / "scene.mjb"))
        disable_model_sameframe_shortcuts(model)
        model.stat.extent = 4.
        model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
        model.light_castshadow[:] = False
        model.mat_reflectance[:] = 0.
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
        if base < 0:
            raise ValueError("场景中找不到 robot/base_link")
        renderer = mujoco.Renderer(model, height=480, width=640)
        data = mujoco.MjData(model)
        try:
            for col, env_id in enumerate(TARGETS):
                row = rows[env_id]
                for key in states:
                    if key.startswith("model_"):
                        dest = getattr(model, key.removeprefix("model_"))
                        dest[:] = states[key][col].reshape(dest.shape)
                count = round(row["duration_s"] / .02)
                start = row["knockdown_at_s"] - 2.
                end = count * .02
                frames = int(np.ceil((end - start) * 30 - 1e-6))
                prefix = "waived-" if env_id in (93, 58) else ""
                name = f"{prefix}{row['magnitude']:g}_{row['knockdown_at_s']:g}s_env{env_id}_{TERRAINS[row['terrain_kind']]}.mp4"

                def frame_at(time_s):
                    idx = min(count - 1, max(0, round(time_s / .02) - 1))
                    data.qpos[:] = states["qpos"][idx, col]
                    data.qvel[:] = states["qvel"][idx, col]
                    if model.nmocap:
                        data.mocap_pos[:] = states["mocap_pos"][idx, col]
                        data.mocap_quat[:] = states["mocap_quat"][idx, col]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = data.xpos[base]
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    canvas = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(canvas)
                    g = states["gravity"][idx, col]
                    tilt = np.degrees(max(abs(np.arctan2(-g[1], -g[2])), abs(np.arcsin(np.clip(g[0], -1., 1.)))))
                    draw.rectangle((0, 0, 640, 88), fill="black")
                    draw.text((8, 5), f"env {env_id} | {row['magnitude']:g} rad/s | hit {row['knockdown_at_s']:g}s | {TERRAINS[row['terrain_kind']]}", font=font, fill="white")
                    draw.text((8, 32), f"t={time_s:.2f}s | tilt={tilt:.1f} deg | last5 balance={row['last5_pure_v2_fraction']:.3f}", font=font, fill="white")
                    draw.text((8, 59), "RECORDED RUN | differs from original result.json", font=font, fill="#ffce70")
                    return canvas

                frame_at(row["knockdown_at_s"]).save(output / f"preview-env{env_id}.png")
                with imageio.get_writer(str(output / name), fps=30, codec="libx264", quality=8,
                                        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
                    for frame in range(frames):
                        # 50 Hz 原始状态选最近帧；不重新仿真、不插值或加减速。
                        writer.append_data(np.asarray(frame_at(start + frame / 30.)))
                for j, time_s in enumerate((row["knockdown_at_s"], end)):
                    poster = frame_at(time_s).resize((640, 360))
                    sheet.paste(poster, (640 * j, col * 392 + 32))
                ImageDraw.Draw(sheet).text((8, col * 392 + 5), f"{name} | left: knockdown time; right: last active state", font=font, fill="black")
                probe = json.loads(subprocess.check_output([
                    str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
                    "-of", "json", str(output / name)], text=True))
                stream = probe["streams"][0]
                if (stream["codec_name"] != "h264" or int(stream["nb_read_frames"]) != frames
                        or stream["r_frame_rate"] != "30/1"
                        or stream["width"] < 640 or stream["height"] < 480):
                    raise RuntimeError(f"视频编码检查失败：{name}: {stream}")
                videos.append({"file": name, "env": env_id, "start_s": start,
                               "last_active_s": end, "ffprobe": stream})
                print("VIDEO " + name + " " + json.dumps(stream), flush=True)
        finally:
            renderer.close()
    sheet.save(output / "contact-sheet.png")
    dump(output / "videos.json", videos)
    write_readme(output, rows, videos)


def write_readme(output, rows, videos):
    original = {r["env"]: r for r in json.loads(SOURCE.read_text())["episodes"]}
    lines = ["轨迹不可逐位复现（verification.json为证据），视频展示的是本次运行的轨迹。", "", "# model_4850 本次运行离线回放", "",
             "视频为640×480、30 fps、H.264；从预定击倒前2秒到最后一个存活状态。",
             "50 Hz GPU轨迹以最近状态采样至30 fps；CPU只做 mj_forward，无物理重积分。",
             "本次7个样本均存活20秒，包括env58；其在原始产物中约15.02秒物理终止。",
             "末5秒纯平衡占比≥0.95才表示持续平衡；recovered另含高度条件与恢复期限，表示曾恢复。",
             "倾角沿用 assess 的 max(|roll|, |pitch|)，不改变测量定义。", "",
             "| 视频 | 应该看到什么 | 时长 / 帧数 |", "|---|---|---|"]
    for item in videos:
        probe = item["ffprobe"]
        row = rows[item["env"]]
        if row["env"] in (93, 58):
            description = "原豁免对照：这次运行里站起来了，是运行间不确定性的直接样本。"
        elif row["env"] == 311:
            description = f"过程最大倾角{row['max_tilt_deg']:.1f}°后回到近竖直，末段未持续平衡。"
        elif row["env"] == 318:
            description = "末段纯平衡达标，但未在期限内满足含高度的曾恢复条件。"
        elif row["last5_pure_v2_fraction"] >= .95:
            description = "曾恢复，本次运行末段持续平衡达标，与原始失败轨迹不同。"
        else:
            description = "曾恢复、末态近水平，但末段未持续平衡。"
        description += f"末5秒纯平衡占比{row['last5_pure_v2_fraction']:.3f}，末态倾角{row['final_tilt_deg']:.2f}°。"
        lines.append(f"| [{item['file']}]({item['file']}) | {description} | {probe['duration']} s / {probe['nb_read_frames']} |")
    lines += ["", "## 原始产物与本次运行对照", "",
              "两栏依次列 recovered／末5秒纯平衡占比／末态倾角°／终止方式；time_out表示满20秒。", "",
              "| env | 原始产物 | 本次运行（视频） |", "|---|---|---|"]
    for env_id in TARGETS:
        values = []
        for row in (original[env_id], rows[env_id]):
            values.append(f"{str(row['recovered']).lower()} / {row['last5_pure_v2_fraction']:.6f} / {row['final_tilt_deg']:.6f} / {row['termination']}")
        lines.append(f"| {env_id} | " + " | ".join(values) + " |")
    lines += ["", "7个样本均未通过与原产物的1e-3复现核对，verification.json保留原失败结论，未覆盖。",
              "本次仅渲染既有recorded-states.npz与scene.mjb，未重跑评估。",
              "contact-sheet.png每行左图为预定击倒时刻，右图为最后存活状态。",
              "视频30 fps不包含终点右侧的新一帧；最后显示的原始状态可能距回合终点最多40 ms，末态截图单独取最后一拍。",
              "reproduced-episodes.json保留完整384环境本次评估结果；recorded-states.npz与scene.mjb可再次离线渲染。", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-only", action="store_true", help="仅渲染已记录的本次运行；保留原复现失败证据，不再评估")
    parser.add_argument("--ffprobe", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(SOURCE.read_text())
    if hashlib.sha256(Path(reference["checkpoint"]).read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise SystemExit("checkpoint SHA256不符")
    if args.render_only:
        saved = json.loads((OUTPUT / "reproduced-episodes.json").read_text())
        rows = saved["episodes"]
        actual = {r["env"]: r for r in rows}
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        rows, extras = record(args, OUTPUT, reference)
        actual = verify(OUTPUT, reference, rows, extras["weight_n"])
    render(OUTPUT, actual, args.ffprobe)


if __name__ == "__main__":
    main()
