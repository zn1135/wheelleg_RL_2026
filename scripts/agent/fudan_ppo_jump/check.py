"""20 环境平地 GPU 预检，实际走完整 PPO 数据流，在两个 Adam 前截住更新。"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import mujoco
import numpy as np
import torch
from scipy.optimize import brentq
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import load_model, solve_standing_joint_positions, closed_chain_residual
from ct_mjlab.envs.chuanliantui.huananhu_poses import _front_kinematics
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import TASK, TUCK, EXTEND, jump_config, recipe_record
from .scoring import check_scoring
from .media import render


def leg_geometry():
    model = load_model()
    data = mujoco.MjData(model)
    bounds = model.joint('lf1').range
    result = {'front_chain_range_m': [_front_kinematics(model,data,k)[0] for k in bounds]}
    for target in (TUCK, EXTEND):
        knee = brentq(lambda k: _front_kinematics(model,data,k)[0]-target, *bounds)
        _, hip = _front_kinematics(model,data,knee)
        pose = solve_standing_joint_positions(model,front_joint_targets={'left':(hip,knee),'right':(-hip,-knee)})
        for name, value in pose.items():
            joint = model.joint(name)
            assert not joint.limited or joint.range[0] <= value <= joint.range[1]
            data.qpos[joint.qposadr[0]] = value
        mujoco.mj_forward(model,data)
        residual = closed_chain_residual(model,data)
        lengths = [np.linalg.norm((data.body(side+'wheel').xpos-data.joint(side+'0').xanchor)[[0,2]])
                   for side in ('lf','rf')]
        assert residual < 1e-6 and np.allclose(lengths,target,atol=1e-6)
        result[str(target)] = dict(left_right_m=lengths, max_loop_error_m=residual)
    return result


def run(output, video, landing=False, resume=None, stop_sink=False, preload=False, peak=False, peak_additive=False):
    output.mkdir(parents=True, exist_ok=False)
    geometry = leg_geometry()
    negative = check_scoring()
    torch.manual_seed(1)
    preload = preload or peak or peak_additive
    landing = landing or stop_sink or preload
    cfg = jump_config(landing=landing,stop_sink=stop_sink,preload=preload,peak=peak,peak_additive=peak_additive)
    peak = peak or peak_additive
    recipe = recipe_record(cfg)
    env, _ = task_registry.make_env(TASK,num_envs=20,env_cfg=cfg)
    try:
        with TemporaryDirectory() as tmp:
            runner, _ = task_registry.make_alg_runner(env,TASK,log_dir=Path(tmp),resume=resume)
            from scripts.agent.fudan_ppo_step.remote_training import same_state
            from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
            runner.save('before.pt')
            before = torch.load(Path(tmp)/'before.pt',map_location='cpu',weights_only=False)
            inherited = ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict',
                         'torch_rng','cuda_rng','iterations','transitions','primary_updates','encoder_updates')
            if resume:
                source = torch.load(resume,map_location='cpu',weights_only=False)
                assert all(same_state(before[k],source[k]) for k in inherited)
            initial = {k:v.clone() for k,v in runner.policy.state_dict().items()}
            ids = torch.arange(20,device=env.device)
            env.reset_idx(ids)
            initial_bottom = env.wheel_bottom()
            assert (initial_bottom.min(1).values-.002).abs().max() < 2e-5
            observed_contact = torch.zeros_like(env.last_contact)
            flight_samples = 0
            peak_samples = 0
            active_landing = {name:0 for name in cfg.rewards.scales if name.startswith('landing_')}
            with torch.inference_mode():
                obs, critic, history = runner.initialize_rollout()
                frames = [env.env.sim.data.qpos.cpu().numpy().copy()]
                for i in range(48):
                    actions = runner.alg.act(obs, history, critic)
                    if peak:
                        previous_peak = env.peak_clearance.clone()
                    obs, critic, rewards, done, info, history = env.step(actions)
                    assert (obs.shape,critic.shape,history.shape,actions.shape)==((20,25),(20,141),(20,125),(20,6))
                    assert all(t.isfinite().all() for t in (obs,critic,history,rewards,actions))
                    terms = info['terms']
                    assert set(terms) == set(env.cfg.rewards.scales)
                    assert all(t.abs().max() <= .025001 for t in terms.values())
                    assert torch.allclose(rewards,torch.stack(list(terms.values())).sum(0))
                    observed_contact |= env.ground_contacts()[0]
                    flight_samples += int((terms['flight']>0).sum())
                    if peak:
                        score = terms['wheel_peak_gain']
                        positive = (score > 0) & ~done
                        expected = (env.peak_clearance.square() - previous_peak.square()).clamp_min(0.)
                        assert (score >= 0).all()
                        assert (env.in_jump[positive] & (env.landing.buffer[positive] == 0)).all()
                        assert torch.allclose(score[positive],
                            (expected[positive]*cfg.rewards.scales['wheel_peak_gain']).clamp_max(2.5)*env.dt)
                        peak_samples += int(positive.sum())
                    runner.alg.process_env_step(rewards,done,info,obs)
                    if (i+1)%4==0:
                        frames.append(env.env.sim.data.qpos.cpu().numpy().copy())
                runner.alg.compute_returns(torch.cat((critic,runner.policy.encode(history)),-1))
                if landing:
                    for i in range(152):
                        actions,_ = runner.policy.act_inference(obs,history)
                        obs, _, reward, _, info, history = env.step(actions)
                        assert torch.isfinite(reward).all()
                        for name in active_landing:
                            active_landing[name] += int(info['terms'][name].ne(0).sum())
                        if (i+1)%4==0:
                            frames.append(env.env.sim.data.qpos.cpu().numpy().copy())
                    assert active_landing['landing_reach'] and active_landing['landing_upright']
                    assert active_landing['landing_compression'] and active_landing['landing_support']
                    if preload:
                        for name in ('landing_motor_target','landing_drive','landing_first_drive','landing_early_length'):
                            assert active_landing[name],name
                    if stop_sink or preload:
                        assert active_landing['landing_clearance'] and active_landing['landing_sink']
                        assert active_landing['landing_support_loss']
            assert observed_contact.all() and flight_samples > 0
            if peak:
                assert peak_samples > 0 and env.highest_landed_peak.item() > 0.
            attempts = {}
            def probe(optimizer, counter):
                grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
                assert grads and all(g.isfinite().all() for g in grads)
                attempts[counter] = attempts.get(counter,0)+1
            runner.alg.optimizer_step = probe
            losses = runner.alg.update()
            assert np.isfinite(losses).all() and attempts == dict(primary_updates=20,encoder_updates=20)
            assert all(torch.equal(initial[k],v) for k,v in runner.policy.state_dict().items())

            runner.save('interface-test.pt')
            saved = torch.load(Path(tmp)/'interface-test.pt',map_location='cpu',weights_only=False)
            assert saved['task']==TASK and saved['jump_recipe']==recipe['version']
            for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
                assert same_state(saved[key]['state'],before[key]['state'])
            assert saved['primary_updates']==before['primary_updates']
            assert saved['encoder_updates']==before['encoder_updates']
            runner.writer.close()
            if video:
                render(env.env.sim.mj_model,frames,output/'precheck-20.mp4',grid=True)
            proof = dict(task=TASK, recipe=recipe['version'], passed=True, backend='mujoco_warp_gpu', worlds=20,
                config=recipe, rollout_steps=48, physics_steps=2010 if landing else 490,
                landing_term_samples=active_landing, resume_sha256=sha(resume) if resume else None, optimizer_updates=0, backward_batches=attempts,
                initial_min_wheel_clearance_m=initial_bottom.min(1).values.tolist(),
                airborne_reward_samples=flight_samples, geometry=geometry, negative_examples=negative,
                video='precheck-20.mp4' if video else None)
            if peak:
                proof.update(peak_reward_samples=peak_samples, highest_landed_peak_m=env.highest_landed_peak.item())
            (output/'precheck.json').write_text(json.dumps(proof,indent=2)+'\n')
            print(json.dumps(proof))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--jump-landing',action='store_true')
    parser.add_argument('--jump-preload',action='store_true')
    parser.add_argument('--jump-peak',action='store_true')
    parser.add_argument('--jump-peak-additive',action='store_true')
    parser.add_argument('--jump-stop-sink',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    run(args.output,args.video,args.jump_landing,args.resume,args.jump_stop_sink,args.jump_preload,
        args.jump_peak,args.jump_peak_additive)
