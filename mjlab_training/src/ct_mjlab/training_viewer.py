"""Observe PPO's own environment without advancing or editing its simulation."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import math
import signal
import threading
import time

import mujoco
import mujoco.viewer
from .config import STEP_DT
from mjlab.viewer.model_sync import (
    VIEWER_MODEL_FIELDS, disable_model_sameframe_shortcuts,
)


@dataclass(frozen=True)
class ViewerOptions:
    fps: float = 30.
    env_idx: int = 0
    num_envs: int = 20

    def validate(self, num_envs):
        if not math.isfinite(self.fps) or not 0. < self.fps <= 120.:
            raise ValueError('viewer-fps must be finite and between 0 and 120')
        if not 0 <= self.env_idx < num_envs:
            raise ValueError(f'viewer-env must be between 0 and {num_envs - 1}')
        if self.num_envs < 1:
            raise ValueError('viewer-num-envs must be positive')


class NativeTrainingViewer:
    def __init__(self, env, options):
        self.env = env.unwrapped
        options.validate(self.env.num_envs)
        self.options = options
        self.env_idx = options.env_idx
        self.paused = False
        self.rendering = True
        self.stop_requested = False
        self.last_frame = -math.inf
        # Host model fields are changed by per-environment rendering. Never use
        # the simulator's shared host template as the writable viewer model.
        self.model = deepcopy(self.env.sim.mj_model)
        disable_model_sameframe_shortcuts(self.model)
        self.data = mujoco.MjData(self.model)
        self.other_data = mujoco.MjData(self.model)
        self.perturb = mujoco.MjvPerturb()
        self.handle = None
        self._render_threads = []
        self._wake = threading.Event()
        self.camera_origin = None

    def open(self):
        self.copy_state()
        previous_threads = set(threading.enumerate())
        try:
            self.handle = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.on_key,
                show_left_ui=False, show_right_ui=False)
        finally:
            # MuJoCo's Handle.close() only requests asynchronous shutdown.
            # Keep and join its render thread before Python/CUDA teardown.
            self._render_threads = [thread for thread in threading.enumerate()
                                    if thread not in previous_threads
                                    and '_launch_internal' in thread.name]
        if self.handle is None:
            raise RuntimeError('Cannot open native training viewer; check DISPLAY and XAUTHORITY')
        with self.handle.lock():
            self.handle.opt.geomgroup[3:] = 0
            self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            rows, cols = self.grid_shape()
            spacing = self.env.scene.env_spacing
            self.handle.cam.lookat[:] = self.env.scene.env_origins[self.env_idx].cpu().numpy()
            self.camera_origin = self.handle.cam.lookat.copy()
            self.handle.cam.lookat[:] += ((rows - 1) * spacing / 2, (cols - 1) * spacing / 2, .3)
            self.handle.cam.distance = max(2., max(rows, cols) * spacing * 1.5)
            self.handle.cam.azimuth = 135.
            self.handle.cam.elevation = min(-40., -40.)

    def on_key(self, key):
        if key == 32:  # Space: pause sampling, keeping orbit/zoom available.
            self.paused = not self.paused
        elif key == 86:  # V: keep training while skipping display updates.
            self.rendering = not self.rendering
            self.last_frame = -math.inf
        elif key in (91, 93):  # [ / ]: previous/next sampled environment.
            self.env_idx = (self.env_idx + (-1 if key == 91 else 1)) % self.env.num_envs
        self._wake.set()

    def is_running(self):
        return self.handle is not None and self.handle.is_running()

    def visible_envs(self):
        return [(self.env_idx + i) % self.env.num_envs
                for i in range(min(self.options.num_envs, self.env.num_envs))]

    def grid_shape(self):
        count = min(self.options.num_envs, self.env.num_envs)
        cols = math.ceil(math.sqrt(count))
        return math.ceil(count / cols), cols

    def snapshot(self, indices):
        sim = self.env.sim
        states = {name: getattr(sim.data, name)[indices].cpu().numpy()
                  for name in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat', 'xfrc_applied')}
        fields = {name: getattr(sim.model, name)[indices].cpu().numpy()
                  for name in sim.expanded_fields & VIEWER_MODEL_FIELDS}
        times = self.env.episode_length_buf[indices].cpu().numpy() * STEP_DT
        return states, fields, times

    def copy_state(self, snapshot=None, row=0, data=None):
        states, fields, times = self.snapshot([self.env_idx]) if snapshot is None else snapshot
        data = self.data if data is None else data
        for name, values in fields.items():
            getattr(self.model, name)[:] = values[row]
        for name, values in states.items():
            getattr(data, name)[:] = values[row]
        data.time = float(times[row])
        mujoco.mj_forward(self.model, data)
        return self.env_idx

    def draw_scene(self):
        indices = self.visible_envs()
        snapshot = self.snapshot(indices)
        origins = self.env.scene.env_origins[indices].cpu().numpy()
        if self.camera_origin is not None:
            self.handle.cam.lookat[:] += origins[0] - self.camera_origin
            self.camera_origin = origins[0].copy()
        scene = self.handle.user_scn
        if (len(indices) - 1) * self.model.ngeom > scene.maxgeom:
            raise ValueError('Too many display geometries; lower --viewer-num-envs')
        scene.ngeom = 0
        _, cols = self.grid_shape()
        spacing = self.env.scene.env_spacing
        for row in range(1, len(indices)):
            self.copy_state(snapshot, row, self.other_data)
            start = scene.ngeom
            mujoco.mjv_addGeoms(self.model, self.other_data, self.handle.opt, self.perturb,
                               mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
            offset = origins[0] - origins[row]
            offset += (row // cols * spacing, row % cols * spacing, 0.)
            for geom in scene.geoms[start:scene.ngeom]:
                geom.pos[:] += offset
        self.copy_state(snapshot)
        return indices

    def draw(self, runner, force=False):
        if not self.is_running():
            self.stop_requested = True
            return
        if not self.rendering and not force:
            if self.last_frame == -math.inf:
                self.handle.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100.value,
                                       mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                                       'Refresh', 'OFF; V resume; Space pause; training continues')])
                self.last_frame = time.monotonic()
            return
        now = time.monotonic()
        if not force and now - self.last_frame < 1. / self.options.fps:
            return
        self.last_frame = now
        with self.handle.lock():
            indices = self.draw_scene()
            idx = indices[0]
        robot = self.env.scene['robot']
        labels = 'Training\nVisible\nState\nKeys'
        iteration = runner.current_learning_iteration
        title = f'Iteration {iteration}' if iteration >= 0 else 'Preview; no learning'
        values = (f'{title}; env {idx + 1}/{self.env.num_envs}\n'
                  f'{indices[0] + 1} .. {indices[-1] + 1} ({len(indices)})\n'
                  f'{self.data.time:.2f} s; {"PAUSED" if self.paused else "RUNNING"}; '
                  f'refresh {"ON" if self.rendering else "OFF"}\n'
                  'V refresh; Space pause; [ / ] select')
        command_manager = getattr(self.env, 'command_manager', None)
        if command_manager is None:
            mode = self.env.recipe.mode
            slots = mode.observation()[idx].detach().cpu().tolist()
            mode_name = 'RECOVER' if mode.recover[idx] else 'NORMAL'
        else:
            slots = self.env._huananhu_observations.mode()[idx].detach().cpu().tolist()
            mode_name = 'RECOVER' if slots[3] > .5 else 'NORMAL'
        labels += '\nMode / auxiliary target / time\nMode slots'
        values += (f'\n{mode_name}; {slots[5]/5:.2f} m; {slots[6]:.2f} s\n'
                   '[' + ', '.join(f'{value:.2f}' for value in slots) + ']')
        if command_manager is None:
            # Direct VecEnv: show the policy's ordinary command separately from
            # the mode's auxiliary height (which is intentionally zero in normal).
            command_height=float(self.env.obs['actor'][idx,3])/5
            height=float(robot.data.root_link_pos_w[idx,2]-self.env.scene.env_origins[idx,2])
            labels += '\nHeight command / measured'
            values += f'\n{command_height:.3f} / {height:.3f} m'
        if command_manager is not None and 'standup' in command_manager.active_terms:
            state = command_manager.get_term('standup')
            height = getattr(state, 'height_above_ground', robot.data.root_link_pos_w[:, 2])
            labels += '\nStart / height\nXY / yaw speed'
            values += (f'\n{"GROUND" if state.ground_start[idx] else "NORMAL"}; '
                       f'{float(state.command[idx, 2]):.3f} / {float(height[idx]):.3f} m\n'
                       f'{float(robot.data.root_link_lin_vel_b[idx, :2].norm()):.3f} m/s / '
                       f'{float(robot.data.root_link_ang_vel_b[idx, 2]):+.3f} rad/s')
        if self.is_running():
            self.handle.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100.value,
                                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value, labels, values)])
        if self.is_running():
            self.handle.sync()

    def wait_if_paused(self, runner):
        while self.paused and not self.stop_requested:
            self.draw(runner)
            self._wake.wait(.02)
            self._wake.clear()

    def pace(self, started):
        remaining = STEP_DT - (time.monotonic() - started)
        while self.rendering and not self.stop_requested and remaining > 0:
            if not self.is_running():
                self.stop_requested = True
                break
            self._wake.wait(remaining)
            self._wake.clear()
            remaining = STEP_DT - (time.monotonic() - started)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        for thread in self._render_threads:
            thread.join(timeout=10.)
            if thread.is_alive():
                raise RuntimeError('Native viewer render thread did not shut down within 10 seconds')
        self._render_threads.clear()


class _StopAfterIteration(Exception):
    pass


def learn_with_viewer(runner, learn, options, *args, viewer_factory=NativeTrainingViewer, **kwargs):
    """Attach rendering to sampling and stop only after a complete PPO update."""
    viewer = viewer_factory(runner.env, options)
    old_step, old_log = runner.env.step, runner.logger.log
    old_sigint = None
    learning_finished = False

    def step(actions):
        viewer.wait_if_paused(runner)
        started = time.monotonic()
        result = old_step(actions)
        viewer.draw(runner)
        viewer.pace(started)
        return result

    def log(*log_args, **log_kwargs):
        result = old_log(*log_args, **log_kwargs)
        if viewer.stop_requested or not viewer.is_running():
            raise _StopAfterIteration
        viewer.draw(runner)
        return result

    def request_stop(_signum, _frame):
        if viewer.stop_requested:
            raise KeyboardInterrupt
        viewer.stop_requested = True
        print('[viewer] Stopping after the current PPO update; saving checkpoint.', flush=True)

    try:
        viewer.open()
        viewer.draw(runner, force=True)
        runner.env.step, runner.logger.log = step, log
        if threading.current_thread() is threading.main_thread():
            old_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, request_stop)
        print('[viewer] Live training window: V refresh, Space pause, [ / ] select env; close to save & stop.', flush=True)
        try:
            result = learn(*args, **kwargs)
            learning_finished = True
            return result
        except _StopAfterIteration:
            if runner.logger.log_dir is not None:
                checkpoint = Path(runner.logger.log_dir) / f'model_{runner.current_learning_iteration}.pt'
                runner.save(str(checkpoint))
                print(f'[viewer] Saved {checkpoint}', flush=True)
    finally:
        runner.env.step, runner.logger.log = old_step, old_log
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
        try:
            viewer.close()
        finally:
            if not learning_finished:
                runner.logger.stop_logging_writer()


def viewer_runner_class(base_runner, options):
    class ViewerRunner(base_runner):
        def learn(self, *args, **kwargs):
            try:
                return learn_with_viewer(self, super().learn, options, *args, **kwargs)
            except BaseException:
                self.env.close()
                raise

    return ViewerRunner
