"""冻结 Jump，在无横条平地逐毫秒测量双轮净空和落地后一秒支撑。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import JUMP_TICKS, LANDING_RECIPES, jump_config
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from .evaluate import COMMAND, SEED, STEPS, WORLDS, TRIGGER, bar_distance, policies
from .handoff import LandingHandoff
from .media import render


def intervals(mask):
    edges = np.diff(np.r_[False,mask,False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def events(trace, modes, *, physics_dt=.001):
    """先有地面双轮支撑、向上脱离，再计算整次飞行中较低轮底的最高点。"""
    def ticks(seconds):
        return round(seconds/physics_dt)

    substeps=ticks(.01)
    contact = trace[:, 8:10] > .5
    clean = (trace[:, 10] <= 1.) & (trace[:, 13] < -.9)
    bottom = trace[:, 6:8].min(1)
    support = contact.all(1) & clean
    jumps = []
    for lift, land in intervals(~contact.any(1)):
        # 排除复位掉落、刚出生的弹跳和只有卸载读数的微振。
        if lift < ticks(.2) or not modes[lift//substeps] or land-lift < ticks(.002):
            continue
        prep = support[max(0,lift-ticks(.15)):lift]
        if not any(b-a >= ticks(.020) for a,b in intervals(prep)) or not clean[lift-1]:
            continue
        if trace[lift-1:lift+1,19].max() <= 0:
            continue
        nonwheel = np.flatnonzero(trace[lift:land,10] > 1.)
        end = lift+int(nonwheel[0]) if len(nonwheel) else land
        if end <= lift or bottom[lift:end].max() <= .002:
            continue
        peak_tick = lift + int(bottom[lift:end].argmax())
        both = np.flatnonzero(contact[land:land+ticks(.150)+1].all(1))
        settle = land+int(both[0]) if both.size else None
        stable = (settle is not None and settle+ticks(1.) <= len(trace) and not len(nonwheel)
                  and clean[land:settle].all() and support[settle:settle+ticks(1.)].all())
        jumps.append(dict(takeoff_s=lift*physics_dt, airtime_s=(land-lift)*physics_dt,
            peak_lower_wheel_m=float(bottom[peak_tick]), wheel_heights_at_peak_m=trace[peak_tick,6:8].tolist(),
            landed=bool(land < len(trace)), stable_1s=bool(stable),
            full_landing_window=bool(settle is not None and settle+ticks(1.) <= len(trace))))
    return jumps


def summarize(cases):
    all_events = [event for case in cases for event in case['jumps']]
    peaks = np.array([e['peak_lower_wheel_m'] for e in all_events])
    first = [case['jumps'][0] for case in cases if case['jumps']]
    return dict(cases=len(cases), triggered=sum(c['triggered'] for c in cases),
        cases_with_takeoff=len(first), total_jumps=len(all_events),
        first_landing_stable_cases=sum(e['stable_1s'] for e in first),
        all_stable_1s=sum(e['stable_1s'] for e in all_events),
        landed=sum(e['landed'] for e in all_events),
        full_landing_window=sum(e['full_landing_window'] for e in all_events),
        peak_median_m=float(np.median(peaks)) if peaks.size else None,
        peak_max_m=float(peaks.max()) if peaks.size else None,
        cases_reaching_5cm=sum(any(e['peak_lower_wheel_m']>=.05 for e in c['jumps']) for c in cases),
        cases_reaching_10cm=sum(any(e['peak_lower_wheel_m']>=.10 for e in c['jumps']) for c in cases),
        all_cases_raw_max_m=max(c['raw_highest_lower_wheel_m'] for c in cases))


@torch.inference_mode()
def collect_profile(output, label, walk, jump, *, landing_handoff=False, physics_dt=.001):
    output.mkdir()
    cfg=jump_config()
    policy_dt=cfg.sim.dt*cfg.control.decimation
    pd_dt=cfg.sim.dt*cfg.control.pd_decimation
    substeps=round(policy_dt/physics_dt)
    pd_steps=round(pd_dt/physics_dt)
    if not (np.isclose(physics_dt*substeps,policy_dt) and np.isclose(physics_dt*pd_steps,pd_dt)):
        raise ValueError('Physics step must evenly divide both controller periods')
    cfg.sim.dt=physics_dt
    cfg.control.decimation=substeps
    cfg.control.pd_decimation=pd_steps
    env = JumpEnv(WORLDS, SEED, 'cuda:0', False, cfg=cfg, record_flat=True)
    try:
        assert env.env.sim.mj_model.geom('terrain').id >= 0 and not env.has_bar
        torch.manual_seed(SEED)
        env.reset_idx(torch.arange(WORLDS, device=env.device))
        env.commands[:] = env.commands.new_tensor(COMMAND)
        obs, _ = env.observations()
        history = obs.repeat(1,5)
        env.history.copy_(history)
        env._history_needs_fill.zero_()
        initial = [a.cpu().numpy().copy() for a in (env.env.sim.data.qpos, env.env.sim.data.qvel, history)]
        start = torch.full((WORLDS,), -1, device=env.device)
        handoff = LandingHandoff(start) if landing_handoff and label == 'walk_jump_walk' else None
        live = torch.ones_like(start,dtype=torch.bool)
        count = np.zeros(WORLDS,dtype=int)
        traces, poses, modes = [], [], []
        for tick in range(STEPS):
            if label == 'jump_only':
                active = live.clone()
            else:
                distance = bar_distance(env)
                start[live & (start<0) & (distance>0) & (distance<TRIGGER)] = tick
                active = handoff.active(tick) if handoff else (start>=0) & (tick-start<JUMP_TICKS)
            action,_ = jump.act_inference(obs,history)
            if label == 'walk_jump_walk':
                walk_obs, walk_hist = obs.clone(),history.reshape(WORLDS,5,25).clone()
                walk_obs[:,6] *= 2/3
                walk_hist[:,:,6] *= 2/3
                walk_action,_ = walk.act_inference(walk_obs,walk_hist.flatten(1))
                action = torch.where(active[:,None],action,walk_action)
            if tick%4 == 0:
                poses.append(env.env.sim.data.qpos.cpu().numpy().copy())
            modes.append(active.cpu().numpy())
            count += live.cpu().numpy()
            env.substeps.clear()
            obs,_,_,done,_,history = env.step(action,auto_reset=False,resample=False)
            if len(env.substeps)!=substeps or env.env.sim.wp_data.overflow.numpy().any():
                raise RuntimeError('Incomplete flat-ground GPU recording')
            rows = torch.stack(env.substeps)
            if handoff:
                handoff.observe(tick,rows,active & live)
            traces.append(rows.cpu().numpy())
            live &= ~done
            if not live.any():
                break
            if done.any():
                env.reset_idx(done.nonzero().flatten())
                env.commands[:] = env.commands.new_tensor(COMMAND)
        traces,poses,modes = np.concatenate(traces),np.asarray(poses),np.asarray(modes)
        cases = []
        for idx, n in enumerate(count):
            trace, mode = traces[:n*substeps,idx],modes[:n,idx]
            assert len(trace) and np.isfinite(trace).all() and not trace[:,20].any()
            jumps = events(trace,mode,physics_dt=physics_dt)
            cases.append(dict(id=idx, triggered=bool(mode.any()), jumps=jumps,
                raw_highest_lower_wheel_m=float(trace[:,6:8].min(1).max())))
            np.savez_compressed(output/f'start{idx:02}.npz',trace=trace,mode=mode,
                qpos0=initial[0][idx],qvel0=initial[1][idx],history0=initial[2][idx],
                poses_25fps=poses[:(n+3)//4,idx])
        summary = summarize(cases)
        (output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
        # 保存最高有效跳跃所在试次；没有有效跳跃则展示固定的第一个试次。
        best = max(range(WORLDS),key=lambda i:max((e['peak_lower_wheel_m'] for e in cases[i]['jumps']),default=0))
        render(env.env.sim.mj_model,poses[:(count[best]+3)//4,best:best+1],output/'replay.mp4')
        return dict(summary,video_case=best)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--landing-handoff',action='store_true',help='Replay an older Jump with touchdown handoff')
    args = p.parse_args()
    walk,jump,saved = policies(args.checkpoint,'cuda:0')
    args.output.mkdir(parents=True,exist_ok=False)
    landing_handoff = args.landing_handoff or saved['jump_recipe'] in LANDING_RECIPES
    result = dict(landing_handoff=landing_handoff,backend='mujoco_warp_gpu', dt=.001, seed=SEED, command=COMMAND,
        checkpoint_sha256=sha(args.checkpoint), iterations=saved['iterations'], optimizer_updates=0)
    for label in ('jump_only','walk_jump_walk'):
        result[label] = collect_profile(args.output/label,label,walk,jump,landing_handoff=landing_handoff)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for i in range(WORLDS):
        with (np.load(args.output/'jump_only'/f'start{i:02}.npz') as a,
              np.load(args.output/'walk_jump_walk'/f'start{i:02}.npz') as b):
            assert all(np.array_equal(a[k],b[k]) for k in ('qpos0','qvel0','history0'))
    print(json.dumps(result))


if __name__=='__main__':
    main()
