#!/usr/bin/env python3
"""Record first-contact getup failures and first-active PD for a frozen checkpoint."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs import task_registry


@torch.inference_mode()
def diagnose(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    cfg,agent=task_registry.get_cfgs('chuanliantui_getup',play=True)
    cfg.auto_reset=False;cfg.seed=43
    env,_=task_registry.make_env('chuanliantui_getup',env_cfg=cfg,train_cfg=agent,num_envs=4,device=args.device)
    raw=env.unwrapped
    try:
        runner,_=task_registry.make_alg_runner(env,'chuanliantui_getup',train_cfg=agent,device=args.device)
        runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location=args.device)
        policy=runner.get_inference_policy(device=args.device)
        saved={k:v.cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        obs,_=raw.reset(seed=43);state=raw.command_manager.get_term('standup');pd=raw.action_manager.get_term('pd')
        assert not pd.action_delay_steps.any()
        with np.load(args.recording/'trajectory.npz') as previous:
            assert np.array_equal(raw.sim.data.qpos.cpu().numpy(),previous['qpos'][0])
        first_active=[None]*4;failure=[None]*4;live=torch.ones(4,dtype=torch.bool,device=args.device)
        peak=torch.zeros(4,device=args.device);records=[];original=pd.apply_actions

        def apply():
            legs=pd._leg_action_ids;hold=pd._ground_hold()
            target=(pd.nominal_positions[:,legs]+pd.cfg.leg_position_scale*pd._control_actions[:,legs]).clamp(
                pd.reference_positions[:,legs]-3.14,pd.reference_positions[:,legs]+3.14)
            error=target-raw.sim.data.qpos[:,pd._motor_q_adr][:,legs]
            original()
            peak.copy_(torch.maximum(peak,torch.where(live,pd.torques[:,legs].abs().amax(-1),0.)))
            for i in range(4):
                if not hold[i] and first_active[i] is None:
                    first_active[i]=dict(policy_tick_end_s=(tick+1)*raw.step_dt,
                        raw_leg_action=pd.raw_action[i,legs].cpu().tolist(),
                        filtered_leg_action=pd._control_actions[i,legs].cpu().tolist(),
                        position_error_rad=error[i].cpu().tolist(),leg_torque_Nm=pd.torques[i,legs].cpu().tolist())

        with patch.object(pd,'apply_actions',apply):
            for tick in range(60):
                action=policy(obs)
                if agent.clip_actions is not None:action=action.clamp(-agent.clip_actions,agent.clip_actions)
                obs,_,terminated,timeout,_=raw.step(action)
                pd_torque_reward=raw.cfg.rewards['torques'].func(raw,name='torques')*raw.cfg.rewards['torques'].weight
                support=-raw.scene['wheel_contact'].data.force[:,:,2]
                gravity_z=raw.scene['robot'].data.projected_gravity_b[:,2]
                for i in (live & terminated).nonzero().flatten().tolist():
                    failure[i]=dict(time_s=(tick+1)*raw.step_dt,has_stood=bool(state.has_stood[i]),
                        inverted=bool(gravity_z[i]>0),airborne_ticks=int(state.wheels_airborne_steps[i]),
                        numerical_failure=bool(state.numerical_failure[i]),torque_reward_before_dt=float(pd_torque_reward[i]))
                records.append(torch.cat((pd.raw_action,pd._control_actions,pd.torques,support,
                    gravity_z[:,None],state.has_stood[:,None],state.wheels_airborne_steps[:,None],
                    live[:,None],pd_torque_reward[:,None]),-1).cpu().numpy())
                live&=~(terminated|timeout)
                if tick<59 and (terminated|timeout).any():
                    # 只复位已结束环境供仿真继续；live已冻结其首回合统计。
                    raw.reset(env_ids=(terminated|timeout).nonzero(as_tuple=False).flatten())
                    obs=raw.get_observations()
        assert all(torch.equal(v.cpu(),saved[k]) for k,v in runner.alg.actor.state_dict().items())
        np.savez_compressed(args.output/'trace.npz',values=np.asarray(records),
            fields=np.asarray([*[f'raw_{i}' for i in range(6)],*[f'control_{i}' for i in range(6)],
                *[f'torque_{i}' for i in range(6)],'support_l','support_r','gravity_z','has_stood',
                'airborne_ticks','first_episode_active','torque_reward_before_dt']))
        report=dict(checkpoint=str(args.checkpoint),steps=60,num_envs=4,seed=43,parameter_updates=0,
            initial_qpos_matches_recording=True,first_active=first_active,first_failure=failure,
            first_episode_peak_leg_torque_Nm=peak.cpu().tolist(),actor_unchanged=True,
            scope='Frozen first 0.6s diagnostic; no causal learning ablation or behavior acceptance')
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Diagnostic cannot train')):
        diagnose(parser.parse_args())
