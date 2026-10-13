"""Canonical64 height scans, native knee rows and paired v2 acceptance; no learning."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo
sys.dont_write_bytecode=True
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import height_knee as h
from ct_mjlab.policy import actor_model
from ct_mjlab.reference_training import huananhu_observations
from ct_mjlab.wheel_balance_assessment import paired_acceptance,STANDARD

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('height_audit_helpers',Path(__file__).with_name('diagnose_chuanliantui_height_response.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
HEIGHTS=(.20,.22,.25,.27,.30,.34)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-normal',type=Path,required=True)
    p.add_argument('--initial-ground',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--heights',type=float,nargs='+',default=list(HEIGHTS))
    args=p.parse_args()
    if len(set(args.heights))!=len(args.heights) or any(x not in HEIGHTS for x in args.heights):
        p.error('Unique authorized height commands only')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    case=saved['infos']['interface']['case']
    infos=h.validate_checkpoint(saved,case)
    reference=json.loads(args.reference.read_text())
    assert reference['checkpoint_sha256']==h.SOURCE_SHA256
    actor=actor_model().to(args.device).eval();actor.load_state_dict(saved['actor_state_dict'],strict=True)
    assert all(torch.isfinite(x).all() for x in actor.state_dict().values())
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(execution_command=shlex.join([sys.executable,*sys.argv]),started=now(),ended=None,
        checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        case=case,completed_updates=infos['completed_updates'],
        learning=False,criteria=h.source.PARAMETERS,command_target_heights=list(args.heights),
        command=dict(height=.30,vx=0.,yaw=0.),num_envs=64,seconds=20.,noise_and_delay=False,
        nominal_physics=True,evaluation_standard=STANDARD,device=args.device,
        gpu=torch.cuda.get_device_name(args.device),versions={n:importlib.metadata.version(n) for n in
            ('torch','mjlab','mujoco','mujoco-warp','warp-lang')},
        simulation_configuration='A baseline play physics/observations with manual commands; no learning',
        initial_inputs_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in
            (args.initial_normal,args.initial_ground)},records={},poses={},fits={},exit_code=None)
    try:
        for pose in ('normal','ground'):
            torch.manual_seed(42)
            cfg=h.source.make_env_cfg('baseline',64,42,play=True)
            cfg.auto_reset=False
            cfg.commands['standup'].ground_start_probability=float(pose=='ground')
            cfg.commands['standup'].zero_velocity_commands=True
            cfg.commands['standup'].height_range=(.30,.30)
            cfg.commands['standup'].resampling_time_range=(1e6,1e6)
            cfg.terminations['failure']=TerminationTermCfg(func=helpers.numerical_probe)
            wrapped=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg,device=args.device),clip_actions=None)
            env=wrapped.unwrapped
            try:
                with np.load(args.initial_normal if pose=='normal' else args.initial_ground) as archive:
                    initial={k:torch.tensor(archive[k],device=args.device) for k in ('qpos','qvel','origins')}
                assert torch.equal(initial['origins'],env.scene.env_origins)
                fingerprint=hashlib.sha256(initial['qpos'].cpu().numpy().tobytes()+initial['qvel'].cpu().numpy().tobytes()).hexdigest()
                assert fingerprint==reference['poses'][pose]['initial_state_sha256']
                if not (args.output/'scene.mjb').exists():
                    mujoco.mj_saveModel(env.sim.mj_model,str(args.output/'scene.mjb'),None)
                state=env.command_manager.get_term('standup')
                for command in args.heights:
                    start=now();env.common_step_counter=0;env._sim_step_counter=0
                    env.reset(seed=42);ids=torch.arange(64,device=args.device);env.scene.reset(ids)
                    env.sim.data.qpos.copy_(initial['qpos']);env.sim.data.qvel.copy_(initial['qvel'])
                    state.sampled[:,:2]=state.command[:,:2]=0.
                    state.sampled[:,2]=state.command[:,2]=command
                    state.ground_command.zero_();state.ground_start.fill_(pose=='ground');state.numerical_failure.zero_()
                    env.sim.forward();env.sim.sense();huananhu_observations.reset_observations(env,ids)
                    env.observation_manager.reset(ids);env.obs_buf=env.observation_manager.compute(update_history=True)
                    obs=wrapped.get_observations()
                    assert torch.equal(env.sim.data.qpos,initial['qpos']) and torch.equal(env.sim.data.qvel,initial['qvel'])
                    active=torch.ones(64,dtype=torch.bool,device=args.device)
                    fractions={};heights=[];knee=[];trajectory=[env.sim.data.qpos[0].cpu().numpy().copy()]
                    with torch.no_grad():
                        for step in range(1000):
                            obs,_,_,_=wrapped.step(torch.where(active[:,None],actor(obs),0.))
                            active &= ~state.numerical_failure
                            for name in ('base_contact','wheel_contact','wheel_ground_probe','nonwheel_ground_probe'):
                                env.scene[name]._invalidate_cache()
                            height=env.scene['robot'].data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
                            flags=helpers.checks(env,active,height)
                            if step>=750:
                                heights.append(height.clone())
                                for name,value in flags.items():
                                    fractions.setdefault(name,torch.zeros(64,device=args.device))
                                    fractions[name] += value & active
                                knee.append({k:v.clone() for k,v in helpers.knee_limit_constraints(env).items()})
                            if abs(command-.30)<1e-9 and (step+1)%5==0 and step+1<1000:
                                trajectory.append(env.sim.data.qpos[0].cpu().numpy().copy())
                            bad=(env.reset_buf & ~env.reset_time_outs).nonzero().flatten()
                            if len(bad):
                                env.reset(env_ids=bad)
                                state.sampled[bad,:2]=state.command[bad,:2]=0.
                                state.sampled[bad,2]=state.command[bad,2]=command
                                obs=wrapped.get_observations()
                    array=torch.stack(heights).cpu().numpy();valid=active.cpu().numpy()
                    stable=((fractions['wheel_balance']>=237.5)&active).cpu().numpy()
                    track=((fractions['height_tracking']>=237.5)&active).cpu().numpy()
                    raw_knee={k:torch.stack([x[k] for x in knee]).cpu().numpy() for k in knee[0]}
                    limit_active=raw_knee['knee_limit_constraint_count']>0
                    moments={k:helpers.distribution(v[:,valid]) for k,v in raw_knee.items()} if valid.any() else {}
                    row=dict(model=case,pose=pose,command_height_m=command,initial_state_sha256=fingerprint,
                        initial_states_restored=True,started=start,ended=now(),numerical_failures=int((~active).sum()),
                        mean_height_m=float(array[:,valid].mean()) if valid.any() else None,
                        pooled_height_std_m=float(array[:,valid].std()) if valid.any() else None,
                        per_env_mean_height_m=array.mean(0).tolist(),stable_balance=int(stable.sum()),
                        stable_samples=np.where(stable)[0].tolist(),height_tracking=int(track.sum()),
                        height_tracking_samples=np.where(track)[0].tolist(),
                        component_fractions={k:(v/250).cpu().tolist() for k,v in fractions.items()},
                        knee_limit_activation_fraction=limit_active[:,valid].mean((0,1)).tolist() if valid.any() else [None,None],
                        knee_limit_generalized_torque_mean_nm=moments['knee_limit_generalized_torque']['mean'] if valid.any() else [None,None],
                        knee_limit_distributions=moments)
                    write(args.output/f'{pose}-h{command:.2f}.json',dict(summary=row,tail_height=array.tolist(),
                        tail_knee_constraints={k:v.tolist() for k,v in raw_knee.items()}))
                    report['records'][f'{pose}:{case}:{command:.2f}']=row
                    if abs(command-.30)<1e-9:
                        report['poses'][pose]=dict(initial_state_sha256=fingerprint,initial_states_restored=True,
                            balanced_samples=row['stable_samples'],numerical_failures=row['numerical_failures'],
                            sustained_last5s=dict(wheel_balance=row['stable_balance'],height_tracking=row['height_tracking']),
                            mean_last5s_height_m=row['mean_height_m'])
                        np.savez_compressed(args.output/f'{case}-{pose}-qpos.npz',qpos=np.array(trajectory),
                                            time_s=np.arange(len(trajectory))*.1)
                    rows=sorted([v for v in report['records'].values() if v['pose']==pose],key=lambda v:v['command_height_m'])
                    if len(rows)>=2 and all(v['mean_height_m'] is not None for v in rows):
                        x=[v['command_height_m'] for v in rows];y=[v['mean_height_m'] for v in rows]
                        common=sorted(set.intersection(*[set(v['stable_samples']) for v in rows]))
                        report['fits'][pose]=dict(slope_m_per_m=float(np.polyfit(x,y,1)[0]),
                            common_balanced_samples=common,common_balanced_count=len(common),
                            common_balanced_slope_m_per_m=float(np.polyfit(x,
                              [np.mean(np.array(v['per_env_mean_height_m'])[common]) for v in rows],1)[0]) if common else None)
                    write(args.output/'evaluation.json',report)
                    print('HEIGHT_SCAN '+json.dumps({k:row[k] for k in ('model','pose','command_height_m',
                        'mean_height_m','stable_balance','height_tracking','numerical_failures',
                        'knee_limit_activation_fraction','knee_limit_generalized_torque_mean_nm')}),flush=True)
            finally:
                wrapped.close()
        if .30 in args.heights:
            report['acceptance']=paired_acceptance(report,reference)
        report['exit_code']=0
    except Exception:
        report['exit_code']=1
        raise
    finally:
        report['ended']=now();write(args.output/'evaluation.json',report)


if __name__=='__main__':
    main()
