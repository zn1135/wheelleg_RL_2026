#!/usr/bin/env python3
"""Paired 40 s nominal/training-condition evaluation; 500 Hz balance, no updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.envs.chuanliantui_stand_v2.observations import get_state
from ct_mjlab.envs.chuanliantui_stand_v2.mdp import (
    TRACKING_COMMAND_CONTRACT, SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT)
from ct_mjlab.envs.chuanliantui_stand_v2.rewards import restore_tracking_reward_cfg, configure_tight_reward
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest, tensor_digest


PHYSICAL_FIELDS = ("gravity_z", "active", "finite", "left_support_N", "right_support_N",
                   "base_force_N", "roll_rad", "pitch_rad", "roll_pitch_speed_rad_s")


def sample_physics(raw):
    record = getattr(raw, "_phase_evaluation", None)
    if record is not None:
        record.sample()
    return torch.zeros(raw.num_envs, device=raw.device)


class Recorder:
    def __init__(self, raw, *, record_physical=False):
        self.raw = raw
        self.root_q = raw.scene["robot"].indexing.free_joint_q_adr
        self.root_v = raw.scene["robot"].indexing.free_joint_v_adr
        self.weight = raw.sim.model.body_mass.sum(-1) * np.linalg.norm(raw.sim.mj_model.opt.gravity)
        self.live = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.finite = self.live.clone()
        self.inverted = ~self.live.clone()
        self.hold = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
        self.final_speed_sum = torch.zeros(raw.num_envs, device=raw.device)
        self.final_base_sum = torch.zeros_like(self.final_speed_sum)
        self.count = 0
        self.trace = (torch.empty((20000, raw.num_envs, len(PHYSICAL_FIELDS)), device=raw.device)
                      if record_physical else None)

    def sample(self):
        raw = self.raw
        q, v = raw.sim.data.qpos[:, self.root_q], raw.sim.data.qvel[:, self.root_v]
        w, x, y, z = q[:, 3:].unbind(-1)
        gx, gy, gz = 2*(w*y-x*z), -2*(y*z+w*x), -1+2*(x*x+y*y)
        roll = torch.atan2(-gy, -gz).abs()
        pitch = gx.clamp(-1., 1.).asin().abs()
        support = -raw.scene["wheel_contact"].data.force[:, :, 2]
        base = raw.scene["base_contact"].data.force.norm(dim=-1).amax(-1)
        finite = (torch.isfinite(raw.sim.data.qpos).all(-1)
                  & torch.isfinite(raw.sim.data.qvel).all(-1)
                  & torch.isfinite(support).all(-1) & torch.isfinite(base))
        self.finite &= finite | ~self.live
        self.inverted |= self.live & (gz > 0)
        balanced = ((support >= (.1*self.weight).clamp_min(5)[:, None]).all(-1)
            & (support.sum(-1) >= .8*self.weight) & (support.sum(-1) <= 1.2*self.weight)
            & (base <= 1.) & (torch.maximum(roll, pitch) <= np.deg2rad(35))
            & (v[:, 3:5].norm(dim=-1) <= .5) & finite & self.live)
        self.hold = torch.where(balanced, self.hold+1, 0)
        if self.trace is not None:
            # 保存实际积分后的子步状态；首行时间为2ms，首回合结束后active保持False。
            self.trace[self.count] = torch.stack((gz, self.live.float(), finite.float(),
                *support.unbind(-1), base, roll, pitch, v[:, 3:5].norm(dim=-1)), -1)
        if self.count >= 17500:  # 最后5秒只报告漂移，不拿移动速度否决站起。
            self.final_speed_sum += v[:, :2].norm(dim=-1)
            self.final_base_sum += base
        self.count += 1


def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    nominal = args.conditions == "nominal"
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2", play=nominal)
    cfg.auto_reset = False
    cfg.metrics["phase_eval"] = MetricsTermCfg(func=sample_physics, per_substep=True, reduce="last")
    env, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        num_envs=args.num_envs, seed=args.seed, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, "chuanliantui_stand_v2",
            train_cfg=agent, device=args.device)
        # 本工具始终验收零速起立。先在原零速配置生成初态，保持与2999的配对随机数顺序。
        metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("infos", {})
        command_contract = metadata.get("command_contract")
        tracking_checkpoint = command_contract in (
            TRACKING_COMMAND_CONTRACT, SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT)
        state = env.unwrapped.command_manager.get_term("standup")
        state.cfg.post_stand_tracking = tracking_checkpoint
        restore_tracking_reward_cfg(state.cfg, metadata)
        if state.cfg.tracking_tight_term:
            from mjlab.managers.reward_manager import RewardManager
            raw = env.unwrapped
            configure_tight_reward(raw.cfg, state.cfg)
            raw.reward_manager = RewardManager(raw.cfg.rewards, raw, scale_by_dt=raw.cfg.scale_rewards_by_dt)
        state.cfg.speed_curriculum = command_contract in (
            SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT)
        if state.cfg.speed_curriculum:
            for key, value in metadata["standup_curriculum"]["speed_curriculum"]["config"].items():
                setattr(state.cfg, key, value)
        state.cfg.yaw_control = command_contract in (YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT)
        state.cfg.position_hold = command_contract == POSITION_COMMAND_CONTRACT
        if state.cfg.yaw_control:
            for key, value in metadata['yaw_control_config'].items():
                setattr(state.cfg, key, value)
        if state.cfg.position_hold:
            for key, value in metadata['position_hold_config'].items():
                setattr(state.cfg, key, value)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
        if tracking_checkpoint:
            state.cfg.post_stand_tracking = False
            state.cfg.speed_curriculum = False
            # 零速姿态对照：先严格按原契约加载，再关闭命令外环，统一vx/yaw为0。
            state.cfg.yaw_control = state.cfg.position_hold = False
            state.speed_curriculum_frozen = True
            ids = torch.arange(env.num_envs, device=env.unwrapped.device)
            state._resample_command(ids)
            state._update_command(ids)
            env.unwrapped.observation_manager.reset()
            env.unwrapped.obs_buf = env.unwrapped.observation_manager.compute(update_history=True)
        policy = runner.get_inference_policy(device=args.device)
        actor_before = tensor_digest(runner.alg.actor.state_dict())
        raw, state = env.unwrapped, env.unwrapped.command_manager.get_term("standup")
        assert raw.step_dt == .01 and raw.physics_dt == .002
        r = Recorder(raw, record_physical=args.record_physical)
        raw._phase_evaluation = r
        action, observation = raw.action_manager.get_term("pd"), get_state(raw)
        initial = dict(qpos=raw.sim.data.qpos.cpu().numpy().copy(),
            qvel=raw.sim.data.qvel.cpu().numpy().copy(), weight=r.weight.cpu().numpy(),
            mass=raw.sim.model.body_mass.cpu().numpy(), friction=raw.sim.model.geom_friction.cpu().numpy(),
            inertia=raw.sim.model.body_inertia.cpu().numpy(), com=raw.sim.model.body_ipos.cpu().numpy(),
            solref=raw.sim.model.geom_solref.cpu().numpy(),
            kp=action.kp_scale.cpu().numpy(), kd=action.kd_scale.cpu().numpy(),
            strength=action.motor_strength.cpu().numpy(), nominal=action._nominal_pos.cpu().numpy(),
            action_delay=action.action_delay_steps.cpu().numpy(),
            observation_delay=observation.delay.lags.cpu().numpy().copy())
        np.savez_compressed(args.output/"initial.npz", **initial)
        if args.record_physical:
            mujoco.mj_saveModel(raw.sim.mj_model, str(args.output/"scene.mjb"), None)
        stood = torch.zeros_like(r.live)
        first_stand = torch.full((raw.num_envs,), float("nan"), device=raw.device)
        first_end = first_stand.clone()
        obs = env.get_observations()
        with torch.inference_mode():
            for tick in range(4000):
                obs, _, done, _ = env.step(policy(obs))
                fresh = r.live & ~stood & state.has_stood
                first_stand[fresh] = (tick+1)*raw.step_dt
                stood |= r.live & state.has_stood
                ended = r.live & done.bool()
                first_end[ended] = (tick+1)*raw.step_dt
                failed = ended & (state._isaac_failure | (tick < 3999))
                r.live &= ~failed
                r.hold[failed] = 0
                if tick < 3999:
                    ids = done.nonzero(as_tuple=False).flatten()
                    if len(ids):
                        raw.reset(env_ids=ids)
                        obs = env.get_observations()
                if (tick+1) % 1000 == 0:
                    print(f"{args.conditions} seed={args.seed}: {(tick+1)/100:.0f}s, ever stood={int(stood.sum())}", flush=True)
        assert r.count == 20000
        assert actor_before == tensor_digest(runner.alg.actor.state_dict())
        balanced = r.live & r.finite & (r.hold >= 500)
        passed = stood & balanced
        array = lambda v: v.detach().cpu().numpy()
        np.savez_compressed(args.output/"episodes.npz", ever_stood=array(stood),
            final_balanced=array(balanced), passed=array(passed), inverted=array(r.inverted),
            finite=array(r.finite), survived=array(r.live), first_stand_s=array(first_stand),
            first_end_s=array(first_end), final_hold_steps=array(r.hold),
            last5_mean_speed=array(r.final_speed_sum/2500), last5_mean_base_force=array(r.final_base_sum/2500))
        report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
            xml_sha256=digest(MODEL_XML),
            scope="40s first episode, deterministic actions; paired comparison, not the separate 20s acceptance",
            conditions=args.conditions, seed=args.seed, num_envs=args.num_envs,
            command_mode="zero-speed get-up evaluation", checkpoint_has_tracking_recipe=tracking_checkpoint,
            duration_s=40, policy_dt_s=.01, physics_dt_s=.002, samples=20000,
            training_noise_delays_randomization_and_pushes=not nominal,
            parameter_updates=0, actor_and_encoder_unchanged=True,
            ever_stood=int(stood.sum()), final_balanced=int(balanced.sum()), passed=int(passed.sum()),
            ever_inverted=int(r.inverted.sum()), finite=int(r.finite.sum()), survived=int(r.live.sum()),
            mean_first_stand_s=float(first_stand[stood].mean()) if stood.any() else None,
            passed_last5_mean_speed=float((r.final_speed_sum[passed]/2500).mean()) if passed.any() else None,
            sources={str(p):digest(p) for p in [Path(__file__), *Path("mjlab_training/src/ct_mjlab").rglob("*.py")]})
        if args.record_physical:
            np.savez_compressed(args.output/"trajectories.npz", physical=array(r.trace), weight=array(r.weight))
            report["physical_fields"] = PHYSICAL_FIELDS
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k != "sources"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", choices=("nominal", "training"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-physical", action="store_true",
                        help="Save actual500Hz first-episode masks, inversion/balance state and scene")
    main(parser.parse_args())
