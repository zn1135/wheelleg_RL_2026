"""固定 20 个初态 × 两档横条；采集和离线验收分别运行。"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import (BAR_DEPTH, BAR_HALF_WIDTH, BAR_X,
    HEIGHTS, JUMP_TICKS, LANDING, LANDING_RECIPES, TASK, TRIGGER, jump_config)
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.train import load_jump_policy
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.utils.task_registry import ROOT
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from scripts.agent.motion.acceptance import envelope
from .handoff import LandingHandoff
from .media import render
from .scoring import score

WALK = ROOT/'artifacts/fudan_ppo_walk/models/20261116_model6838_regions_scan_slope5to20_mu08/model_final.pt'
WALK_SHA = '8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad'
WORLDS, SEED, STEPS = 20, 823, 600
COMMAND = (2., 0., .26)
PROTOCOL = dict(distance_model='horizontal_forward_ray_to_front_face', version='jump-bar-gpu-v1', worlds=WORLDS, seed=SEED, max_steps=STEPS,
    heights_m=list(HEIGHTS), bar_start_m=BAR_X, depth_m=BAR_DEPTH, half_width_m=BAR_HALF_WIDTH,
    trigger_m=TRIGGER, jump_policy_steps=JUMP_TICKS, command=list(COMMAND), physics_dt=.001, policy_dt=.01)

LANDING_PROTOCOL = dict(PROTOCOL, version='jump-bar-gpu-landing-v2',
    jump_policy_steps=None, air_ticks=LANDING['air_ticks'], buffer_ticks=LANDING['buffer_ticks'],
    max_jump_ticks=LANDING['max_jump_ticks'])


def case_id(height, idx):
    return f'bar{round(100*height):02}_start{idx:02}'


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str((a.shape,a.dtype)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def policies(checkpoint, device):
    jump, jump_saved = load_jump_policy(checkpoint, device)
    if sha(WALK) != WALK_SHA:
        raise ValueError('Default 6838 walking model checksum differs')
    walk, walk_saved = load_policy(WALK,device=device)
    if walk_saved['iterations'] != 6838 or walk_saved.get('reset_contract') != RESET_CONTRACT:
        raise ValueError('Unexpected walking baseline')
    if (json.dumps(walk_saved['control'], sort_keys=True) != json.dumps(jump_saved['control'], sort_keys=True)
            or walk_saved['control_timing'] != jump_saved['control_timing']):
        raise ValueError('Policy control contracts differ')
    return walk, jump, jump_saved


def bar_distance(env):
    # 理想测距探头沿车头的水平投影观察前沿，未命中有限宽横条时返回 -1。
    w, x, y, z = env.robot.data.root_link_quat_w.unbind(-1)
    dx = 1 - 2*(y*y + z*z)
    dy = 2*(x*y + w*z)
    scale = (BAR_X - env.position[:, 0]) / dx.clamp_min(1e-6)
    hit_y = env.position[:, 1] + scale*dy
    distance = scale*torch.hypot(dx, dy)
    valid = (dx > 0) & (scale > 0) & (hit_y.abs() <= BAR_HALF_WIDTH)
    return torch.where(valid, distance, -1.)


@torch.inference_mode()
def collect_height(output, height, walk, jump, device, *, landing_handoff=False):
    cfg = jump_config()
    env = JumpEnv(WORLDS,SEED,device,False,cfg=cfg,bar_height=height)
    try:
        model = env.env.sim.mj_model
        bar = model.geom('jump_bar')
        assert np.allclose(bar.size,[BAR_DEPTH/2,BAR_HALF_WIDTH,height/2])
        assert np.allclose(bar.pos,[BAR_X+BAR_DEPTH/2,0,height/2])
        assert (env.env.scene.env_origins == 0).all()
        torch.manual_seed(SEED)
        env.reset_idx(torch.arange(WORLDS,device=device))
        env.commands[:] = env.commands.new_tensor(COMMAND)
        obs, _ = env.observations()
        history = obs.repeat(1,5)
        env.history.copy_(history)
        env._history_needs_fill.zero_()
        initial = [a.cpu().numpy().copy() for a in (env.env.sim.data.qpos,env.env.sim.data.qvel,history)]
        tick_started = torch.full((WORLDS,),-1,device=device)
        handoff = LandingHandoff(tick_started) if landing_handoff else None
        alive = torch.ones(WORLDS,dtype=torch.bool,device=device)
        counts = np.zeros(WORLDS,dtype=int)
        frames, traces, distances, modes, commands = [], [], [], [], []
        for tick in range(STEPS):
            distance = bar_distance(env)
            activate = alive & (tick_started < 0) & (distance > 0) & (distance < TRIGGER)
            tick_started[activate] = tick
            active = handoff.active(tick) if handoff else (tick_started>=0) & (tick-tick_started<JUMP_TICKS)
            # 同一真实传感/动作历史，用各自训练时的速度指令缩放重新编码全部五帧。
            walk_obs, walk_history = obs.clone(), history.reshape(WORLDS,5,25).clone()
            walk_obs[:,6] *= 2/3
            walk_history[:,:,6] *= 2/3
            walk_action, _ = walk.act_inference(walk_obs,walk_history.flatten(1))
            jump_action, _ = jump.act_inference(obs,history)
            if tick%4 == 0:
                frames.append(env.env.sim.data.qpos.cpu().numpy().copy())
            counts += alive.cpu().numpy()
            distances.append(distance.cpu().numpy())
            modes.append(active.cpu().numpy())
            commands.append(env.commands.cpu().numpy().copy())
            env.substeps.clear()
            obs, _, _, done, _, history = env.step(torch.where(active[:,None],jump_action,walk_action),
                                                   auto_reset=False,resample=False)
            assert len(env.substeps)==10
            if env.env.sim.wp_data.overflow.numpy().any():
                raise RuntimeError('GPU contact/constraint capacity overflow')
            rows = torch.stack(env.substeps)
            if handoff:
                handoff.observe(tick,rows,active & alive)
            traces.append(rows.cpu().numpy())
            alive &= ~done
            if not alive.any():
                break
            if done.any():
                # 已失败的世界退出计分；复位只为其余世界完成采集。
                env.reset_idx(done.nonzero().flatten())
                env.commands[:] = env.commands.new_tensor(COMMAND)
        trace = np.concatenate(traces)
        modes, distances, commands, frames = map(np.asarray,(modes,distances,commands,frames))
        reports, initial_ids = {}, []
        videos = {}
        examples = set()
        for idx, count in enumerate(counts):
            key = case_id(height,idx)
            qpos,qvel,hist = (a[idx] for a in initial)
            initial_ids.append(digest_arrays(qpos,qvel,hist))
            path = output/(key+'.npz')
            state, mode, distance = trace[:count*10,idx], modes[:count,idx], distances[:count,idx]
            poses = frames[:(count+3)//4,idx]
            np.savez_compressed(path,trace=state,mode=mode,distance=distance,
                command=commands[:count,idx],qpos0=qpos,qvel0=qvel,history0=hist,poses_25fps=poses)
            outcome = score(state,mode,distance,height,landing_handoff=landing_handoff)
            reason = outcome['reason']
            video = None
            if reason not in examples:
                video = key+'.mp4'
                render(model,poses[:,None,:],output/video)
                examples.add(reason)
                videos[reason] = video
            reports[key] = dict(sha256=sha(path), initial_sha256=initial_ids[-1], video=videos[reason])
        return reports, initial_ids
    finally:
        env.close()


def collect(checkpoint, output, device, *, landing_handoff=False):
    if not device.startswith('cuda'):
        raise ValueError('Use the MuJoCo Warp CUDA backend')
    walk,jump,saved = policies(checkpoint,device)
    output.mkdir(parents=True, exist_ok=False)
    landing_handoff = landing_handoff or saved['jump_recipe'] in LANDING_RECIPES
    protocol = LANDING_PROTOCOL if landing_handoff else PROTOCOL
    record = dict(task=TASK, protocol=protocol, backend='mujoco_warp_gpu', device=device,
        gpu=torch.cuda.get_device_name(device), walk_sha256=sha(WALK), jump_sha256=sha(checkpoint),
        jump_iterations=saved['iterations'], optimizer_updates=0, cases={})
    record['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__).resolve(), Path(__file__).with_name('scoring.py').resolve(),
        Path(__file__).with_name('handoff.py').resolve(),
        ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/fudan_ppo_jump/env.py',
        ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/fudan_ppo_jump/config.py',
        ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')}
    manifest = output/'protocol.json'
    manifest.write_text(json.dumps(record,indent=2))
    for height in HEIGHTS:
        cases,starts = collect_height(output,height,walk,jump,device,landing_handoff=landing_handoff)
        if 'initial_ids' in record and starts != record['initial_ids']:
            raise ValueError('Initial states differ between bar heights')
        record.update(initial_ids=starts)
        record['cases'].update(cases)
        manifest.write_text(json.dumps(record,indent=2))


def assess_recording(directory):
    data = json.loads((directory/'protocol.json').read_text())
    protocol=data['protocol']
    landing_protocols=(LANDING_PROTOCOL,dict(LANDING_PROTOCOL,buffer_ticks=8))
    landing_handoff=protocol in landing_protocols
    if (protocol not in (PROTOCOL,*landing_protocols) or data['backend'] != 'mujoco_warp_gpu' or data['task'] != TASK
            or not data['device'].startswith('cuda') or data['walk_sha256'] != WALK_SHA):
        raise ValueError('Wrong GPU recording protocol or walking baseline')
    rows, by_height = [], {}
    paired = data.get('initial_ids',[])
    for height in HEIGHTS:
        group = []
        for idx in range(WORLDS):
            key = case_id(height,idx)
            entry = data['cases'].get(key)
            path = directory/(key+'.npz')
            verdict = dict(passed=False,reason='missing_recording')
            if path.is_file() and entry:
                if sha(path) != entry['sha256']:
                    raise ValueError('Trace checksum mismatch: '+key)
                with np.load(path) as t:
                    initial_id = digest_arrays(t['qpos0'],t['qvel0'],t['history0'])
                    if len(paired)!=WORLDS or paired[idx] != initial_id or entry['initial_sha256'] != initial_id:
                        raise ValueError('Unpaired initial state: '+key)
                    if t['command'].shape != (len(t['mode']),3) or not np.allclose(t['command'],COMMAND):
                        verdict = dict(passed=False,reason='command_not_delivered')
                    else:
                        verdict = score(t['trace'],t['mode'],t['distance'],height,landing_handoff=landing_handoff,
                                        buffer_ticks=protocol.get('buffer_ticks'))
            crossed = verdict['passed']
            passed = crossed and verdict['post_landing_motion']['passed']
            if crossed and not passed:
                verdict['reason'] = 'post_landing_tracking'
            verdict.update(jump_passed=crossed, passed=passed, video=entry.get('video') if entry else None)
            group.append(dict(sample_id=key, **verdict))
        rows.extend(group)
        by_height[str(height)] = dict(jump_passed=sum(r['jump_passed'] for r in group),
            technical_passed=sum(r['passed'] for r in group), total=len(group))
    result = envelope(TASK,directory,rows,['jump_over_bar'],True,
        rise=dict(status='not_measured'),details=dict(by_height=by_height,
        protocol=protocol['version'],walk_sha256=data['walk_sha256'],jump_sha256=data['jump_sha256'],jump_iterations=data['jump_iterations']))
    result['jump_thresholds_met'] = all(r['jump_passed'] for r in rows)
    result['technical_thresholds_met'] = bool(data['jump_iterations'] > 0 and all(r['passed'] for r in rows))
    result['missing_capabilities'] = ['jump_over_bar'] if any(r['reason']=='missing_recording' for r in rows) else []
    if len(paired) == WORLDS:
        result['cohort_sha256'] = hashlib.sha256(json.dumps([protocol,paired],sort_keys=True).encode()).hexdigest()
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--landing-handoff',action='store_true',help='Replay an older Jump with touchdown handoff')
    args=parser.parse_args()
    collect(args.checkpoint,args.output,args.device,landing_handoff=args.landing_handoff)
