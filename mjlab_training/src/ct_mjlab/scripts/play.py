"""Replay a checkpoint in the play configuration (same flow as wheel_legged_gym/scripts/play.py)."""
import os
from pathlib import Path

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.huananhu_randomization import knockdown_root_velocity
from ct_mjlab.envs.chuanliantui.multimode_env_cfg import KNOCKDOWN_ANGULAR_SPEED_RANGE
from ct_mjlab.utils.helpers import checkpoint_arg, get_args


def periodic_knockdown_policy(env, policy, every_s, speed_range=KNOCKDOWN_ANGULAR_SPEED_RANGE):
    """Wrap a policy so a standing robot is tipped over every ``every_s`` seconds.

    The play configuration registers no disturbances, so this is the only way to
    watch the mid-episode recover path in a viewer. It uses the explicit check's
    "already standing for 0.5 s" gate, so the robot gets to stand up before it is
    knocked down again; a knockdown that arrives while it is still down is
    dropped rather than queued.

    ``speed_range`` defaults to the shared manual-check range, which this path
    used to omit -- and the function's old default (2-3.5 rad/s) tipped 0 of 32
    standing robots, so the viewer silently showed no knockdowns at all.
    """
    raw = env.unwrapped
    elapsed = 0.0

    def wrapped(obs):
        nonlocal elapsed
        elapsed += raw.step_dt
        if elapsed >= every_s:
            elapsed = 0.0
            knockdown_root_velocity(raw, None, angular_speed_range=tuple(speed_range))
        return policy(obs)

    return wrapped


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, play=True)
    if args.seed is not None:
        env_cfg.seed = args.seed
    if args.smp_prior is not None:
        from ct_mjlab.smp.ppo import configure_smp, policy_observation_contract, checkpoint_smp_weight, checkpoint_bc_finetune
        configure_smp(env_cfg, train_cfg, args.smp_prior, play=True,
                      policy_contract=policy_observation_contract(args.checkpoint),
                      reward_weight=checkpoint_smp_weight(args.checkpoint),
                      bc_finetune=checkpoint_bc_finetune(args.checkpoint))
    env, _ = task_registry.make_env(
        args.task, env_cfg=env_cfg, train_cfg=train_cfg, play=True,
        num_envs=args.num_envs, device=args.device)
    try:
        if args.checkpoint is None:
            shape = env.unwrapped.action_space.shape
            policy = lambda obs: torch.zeros(shape, device=env.unwrapped.device)  # noqa: E731
        else:
            runner, _ = task_registry.make_alg_runner(env, args.task, train_cfg=train_cfg, device=args.device)
            runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
        if args.knockdown_every:
            policy = periodic_knockdown_policy(
                env, policy, args.knockdown_every, tuple(args.knockdown_speed))
        viewer = args.viewer
        if viewer == 'auto':
            viewer = 'native' if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') else 'viser'
        if viewer == 'native':
            import mujoco
            from mjlab.viewer import NativeMujocoViewer

            class CommandViewer(NativeMujocoViewer):
                def _set_status_overlay(self, handle):
                    status = self.get_status()
                    capped = ' [CAPPED]' if status.capped else ''
                    # Read the selected robot's current command in physical units.
                    vx, yaw, height = self.env.unwrapped.command_manager.get_term(
                        'standup').command[self.env_idx].detach().cpu().tolist()
                    labels = 'Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT\nCMD vx (m/s)\nCMD yaw (rad/s)\nCMD height (m)'
                    values = (
                        f'{self.env_idx + 1}/{self.env.num_envs}\n{status.step_count}\n'
                        f'{"PAUSED" if status.paused else "RUNNING"}{capped}\n'
                        f'{status.speed_label}\n{status.target_realtime:.2f}x\n'
                        f'{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)\n'
                        f'{vx:+.3f}\n{yaw:+.3f}\n{height:.3f}')
                    handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT.value, labels, values))

            CommandViewer(env, policy).run()
        else:
            from mjlab.viewer import ViserPlayViewer
            if args.viser_port is None:
                ViserPlayViewer(env, policy).run()
            else:
                # viser defaults to port 8080; pass our own server so the viewer
                # can be moved off a port something else already holds.
                import viser
                server = viser.ViserServer(label='mjlab', port=args.viser_port)
                print(f'viser viewer serving on http://localhost:{args.viser_port}')
                ViserPlayViewer(env, policy, viser_server=server).run()
    finally:
        env.close()


def main():
    args = get_args(__doc__, defaults={'num_envs': 1}, extra=(
        checkpoint_arg(),
        (('--smp-prior',), dict(type=Path, default=None, help='Use the SMP PPO profile and its normal ground starts')),
        (('--viewer',), dict(choices=('auto', 'native', 'viser'), default='auto')),
        (('--viser-port',), dict(type=int, default=None,
                                 help='port for the viser viewer (default: viser picks its own, 8080)')),
        (('--knockdown-every',), dict(type=float, default=None,
                                      help='seconds between knockdowns of a standing robot')),
        (('--knockdown-speed',), dict(type=float, nargs=2,
                                      default=KNOCKDOWN_ANGULAR_SPEED_RANGE,
                                      metavar=('MIN', 'MAX'),
                                      help='knockdown angular impulse range rad/s')),
    ))
    play(args)


if __name__ == '__main__':
    main()
