"""在 MuJoCo 窗口逐帧播放已保存的 GPU sim2sim 姿态。"""
import argparse
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np


def run(manifest_path, initial_clip, seconds):
    manifest=json.loads(manifest_path.read_text())
    clips=manifest['items']
    keys=SimpleQueue()
    index=initial_clip
    ended_at=None if seconds is None else time.monotonic()+seconds
    while index is not None:
        clip=clips[index]
        model=mujoco.MjModel.from_binary_path(str(manifest_path.parent/clip['model']))
        state=mujoco.MjData(model)
        with np.load(clip['trace']) as saved:
            poses=saved['poses_25fps']
            modes=saved['mode']
        assert len(poses)==clip['frames'] and poses.shape[1]==model.nq
        assert np.isfinite(poses).all()
        state.qpos[:]=poses[0]
        mujoco.mj_forward(model,state)
        before=set(threading.enumerate())
        handle=mujoco.viewer.launch_passive(model,state,key_callback=keys.put,
                                            show_left_ui=False,show_right_ui=False)
        render_thread=next((t for t in set(threading.enumerate())-before if '_launch_internal' in t.name),None)
        print(f"Opened clip {index+1}: {clip['label']}",flush=True)
        frame=0
        paused=False
        selection=None
        frame_deadline=time.monotonic()
        try:
            while handle.is_running():
                if ended_at is not None and time.monotonic()>=ended_at:
                    return
                try:
                    key=keys.get_nowait()
                except Empty:
                    key=None
                if key is not None and ord('1')<=key<=ord('4'):
                    selection=key-ord('1')
                    break
                if key in (ord('R'),ord('r')):
                    frame=0
                if key==32:
                    paused=not paused
                with handle.lock():
                    state.qpos[:]=poses[frame]
                    state.time=frame/25
                    mujoco.mj_forward(model,state)
                    handle.cam.azimuth=105.
                    handle.cam.elevation=-12.
                    handle.cam.distance=2.
                    handle.cam.lookat[:]=state.qpos[:3]+[.15,0,.08]
                policy='JUMP' if modes[min(frame*4,len(modes)-1)] else 'WALK'
                handle.set_texts((None,None,
                    f"Jump 5200  |  {clip['label']}\n1: flat 1 ms   2: flat 0.5 ms   3: bar 5 cm   4: bar 10 cm\nSpace: pause   R: restart",
                    f"{state.time:.2f} s   {policy}"))
                handle.sync(state_only=True)
                frame_deadline=max(frame_deadline+.040,time.monotonic())
                time.sleep(max(0.,frame_deadline-time.monotonic()))
                if not paused:
                    frame=(frame+1)%len(poses)
        finally:
            handle.close()
            # The GLFW thread owns the rendering context. Wait for it before
            # releasing the model or opening the next recorded scene.
            if render_thread is not None:
                render_thread.join(timeout=3.)
                if render_thread.is_alive():
                    raise RuntimeError('MuJoCo viewer thread did not close')
        index=selection


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--clip',default=2,type=int,choices=(1,2,3,4))
    p.add_argument('--seconds',type=float)
    a=p.parse_args()
    run(a.manifest,a.clip-1,a.seconds)
