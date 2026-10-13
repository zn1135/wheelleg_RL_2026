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
from types import SimpleNamespace
from ct_mjlab.experiments import recovery_factors as factors
from ct_mjlab.reference_training.model import ACTUATED_JOINTS
h = SimpleNamespace(source=factors, SOURCE_SHA256='706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd')
from ct_mjlab.policy import actor_model
from ct_mjlab.reference_training import huananhu_observations
from ct_mjlab.wheel_balance_assessment import paired_acceptance,STANDARD

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('height_audit_helpers',Path(__file__).with_name('diagnose_chuanliantui_height_response.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
HEIGHTS=(.20,.22,.25,.27,.30,.34)
V11_SHA='62efa60cf3c0884d0abf4a2378d778371135a49c475ed3453ebec08136d22366'
OUT=ROOT/'artifacts/v11-replay-20261013'
WHEELLEG=ROOT.parent/'wheelleg_RL'


def validate_v11(saved, path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==V11_SHA
    infos=saved['infos']
    assert infos['observation_contract']=='chuanliantui-huananhu-flat-35-78-v1'
    assert infos['control_contract'].endswith('sourcexml-spring90mm-v10')
    assert infos['huananhu_training_iteration']==4000 and saved['iter']==999
    return infos


def make_replay_cfg(n, play):
    cfg=factors.make_env_cfg('baseline',n,42,play=play)
    cfg.actions['pd'].policy_joint_order=ACTUATED_JOINTS
    cfg.commands['standup'].dynamic_modes=False
    cfg.commands['standup'].zero_velocity_commands=True
    cfg.commands['standup'].special_modes_enabled=False
    cfg.commands['standup'].height_range=(.30,.30)
    cfg.commands['standup'].resampling_time_range=(1e6,1e6)
    cfg.auto_reset=False
    return cfg


def actor_from_checkpoint(path, device='cuda:0'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    validate_v11(saved,path)
    actor=actor_model().to(device).eval()
    actor.load_state_dict(saved['actor_state_dict'],strict=True)
    assert all(torch.isfinite(x).all() for x in actor.state_dict().values())
    return actor


def channels():
    rows=[(n,'1; clamp ±100') for n in ('cmd_vx','cmd_vy=0','cmd_yaw_rate')]
    rows.append(('cmd_height','5; clamp [0,1]'))
    rows.extend((f'base_ang_vel_{a}','0.5; clamp ±100') for a in 'xyz')
    rows.extend((f'projected_gravity_{a}','1; clamp ±100') for a in 'xyz')
    order=('lf0','lf00','rf0','rf00','lfwheel','rfwheel')
    rows.extend((f'{n}_position_minus_nominal' if 'wheel' not in n else f'{n}_position=0','1; clamp ±100') for n in order)
    rows.extend((f'{n}_velocity','0.1; clamp ±200') for n in order)
    rows.extend((f'{n}_last_raw_action','1; clamp ±100') for n in order)
    rows.extend((n,'1; fixed '+str(v)) for n,v in zip(
        ('mode_normal',*(f'normal_placeholder_slot_{i}' for i in range(1,7))),
        (1,0,0,0,0,0,0)))
    assert len(rows)==35
    return [dict(index=i,wheelleg_name=n,local_name=n,wheelleg_scale=s,local_scale=s) for i,(n,s) in enumerate(rows)]


def gate():
    from wheelleg_rl.chuanliantui.huananhu_observations import HuananhuObservationState as SourceObservation
    from wheelleg_rl.chuanliantui.v14_observations import V14ObservationCfg as SourceObsCfg
    from wheelleg_rl.chuanliantui.model import standing_motor_positions as source_nominal
    actor=actor_from_checkpoint(OUT/'models/model_999.pt')
    assert actor.mlp[0].weight.shape==(256,35)
    cfg=make_replay_cfg(8,play=True)
    wrapped=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg,device='cuda:0'),clip_actions=None)
    env=wrapped.unwrapped
    record=dict(started=now(),passed=False,checkpoint_sha256=V11_SHA,channels=channels(),
        action_order=list(ACTUATED_JOINTS),observation_joint_order=['lf0','lf00','rf0','rf00','lfwheel','rfwheel'],
        observation_source_note='Source V14ObservationState explicitly reorders all three joint blocks legs first.',
        mapping=[],learning=False)
    try:
        action=env.action_manager.get_term('pd');state=env.command_manager.get_term('standup')
        assert state.control_mode is None
        state.ground_start.zero_();env.episode_length_buf.fill_(10)
        assert torch.allclose(action.nominal_positions[0,list((0,1,3,4))],
                              torch.tensor(source_nominal(),device=env.device),atol=1e-6,rtol=0)
        # Capture targets computed by the real apply_actions, and actual effort outputs.
        captured={}
        def trace(frame,event,arg):
            if frame.f_code is type(action).apply_actions.__code__ and event=='return':
                captured.update({k:frame.f_locals[k].detach().clone() for k in ('target','target_velocity')})
            return trace
        def apply(values):
            action.process_actions(values)
            sys.settrace(trace)
            try:action.apply_actions()
            finally:sys.settrace(None)
            targets=torch.zeros_like(values)
            targets[:,action._leg_action_ids]=captured['target']
            targets[:,action._wheel_action_ids]=captured['target_velocity']
            return targets.clone(),action.torques.clone()
        env.sim.data.qpos[:,action._motor_q_adr]=action.nominal_positions
        env.sim.data.qvel[:,action._motor_v_adr]=0.
        zero=torch.zeros(8,6,device=env.device)
        baseline,base_tau=apply(zero)
        for i,name in enumerate(ACTUATED_JOINTS):
            unit=zero.clone();unit[:,i]=1.
            targets,tau=apply(unit);delta=targets-baseline;dtau=tau-base_tau
            expected=zero.clone();expected[:,i]=10. if 'wheel' in name else .5
            assert torch.allclose(delta,expected,atol=1e-6,rtol=0)
            assert torch.allclose(dtau[:,i],torch.full((8,),1. if 'wheel' in name else 30.,device=env.device),atol=1e-5)
            assert torch.count_nonzero(dtau[:,[j for j in range(6) if j!=i]])==0
            record['mapping'].append(dict(input_dimension=i,motor=name,target_delta=delta[0].cpu().tolist(),
                                          torque_delta_nm=dtau[0].cpu().tolist(),other_motors_unchanged=True))
        # Unique channel values and clipping extremes on the identical physical state.
        env.sim.data.qpos[:,action._motor_q_adr]=action.nominal_positions+torch.tensor([.11,.22,.33,.44,.55,.66],device=env.device)
        env.sim.data.qvel[:,action._motor_v_adr]=torch.tensor([1.,2.,3.,4.,5.,6.],device=env.device)
        action.process_actions(torch.tensor([.1,.2,.3,.4,.5,.6],device=env.device).repeat(8,1))
        state.command[:]=torch.tensor([.7,.8,.30],device=env.device)
        local=huananhu_observations.get_state(env)
        source=SourceObservation(env,SourceObsCfg(noise=False,delay=False))
        probes=[local.raw_sensors(),torch.arange(18,device=env.device,dtype=torch.float32).repeat(8,1),
                torch.full((8,18),300.,device=env.device),torch.full((8,18),-300.,device=env.device)]
        maximum=0.
        for sensors in probes:
            a=torch.cat((local.basic(sensors),local.mode()),-1)
            b=torch.cat((source.basic(sensors),source.mode()),-1)
            maximum=max(maximum,float((a-b).abs().max()));assert torch.equal(a,b)
        huananhu_observations.reset_observations(env,torch.arange(8,device=env.device))
        assert torch.equal(local.actor(),source.actor())
        assert torch.equal(local.mode(),torch.tensor([1.,0.,0.,0.,0.,0.,0.],device=env.device).repeat(8,1))
        record.update(observation_max_abs_error=maximum,mode_slots=local.mode()[0].cpu().tolist(),
                      nominal_positions=action.nominal_positions[0].cpu().tolist(),passed=True)
    finally:
        wrapped.close();record['ended']=now();write(OUT/'interface-gate.json',record)
    print('V11_INTERFACE_GATE '+json.dumps(record),flush=True)


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
    assert json.loads((OUT/'interface-gate.json').read_text())['passed']
    assert args.output.resolve().is_relative_to(OUT.resolve())
    if len(set(args.heights))!=len(args.heights) or any(x not in HEIGHTS for x in args.heights):
        p.error('Unique authorized height commands only')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    case='v11'
    infos=validate_v11(saved, args.checkpoint)
    reference=json.loads(args.reference.read_text())
    assert reference['checkpoint_sha256']==h.SOURCE_SHA256
    actor=actor_model().to(args.device).eval();actor.load_state_dict(saved['actor_state_dict'],strict=True)
    assert all(torch.isfinite(x).all() for x in actor.state_dict().values())
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(execution_command=shlex.join([sys.executable,*sys.argv]),started=now(),ended=None,
        checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        case=case,completed_updates=infos['huananhu_training_iteration'],
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
            cfg=make_replay_cfg(64, play=True)
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
            report['acceptance']['interpretation']='cross-model comparison only; not continuation regression'
        report['exit_code']=0
    except Exception:
        report['exit_code']=1
        raise
    finally:
        report['ended']=now();write(args.output/'evaluation.json',report)


def source_replay():
    """Original v11 fixed-height method, using source assets and source observations.

    The installed mjlab runtime is reused. No source installation or source writes.
    Current source reward coefficients do not affect deterministic inference.
    """
    assert json.loads((OUT/'interface-gate.json').read_text())['passed']
    from wheelleg_rl.chuanliantui.huananhu_env_cfg import make_huananhu_env_cfg
    from wheelleg_rl.chuanliantui.huananhu_observations import reset_observations
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    actor=actor_from_checkpoint(OUT/'models/model_999.pt')
    report=dict(started=now(),learning=False,runtime=sys.executable,
        source_assets=True,source_observations=True,seed=42,num_envs=64,seconds=20.,
        metric_window='t > 5 s, first live episode (matches archived fixed_height.py)',
        archived_height_m=[.2725,.3085,.3502],archived_within3cm_fraction=[1.,1.,1.],records=[])
    for command in (.27,.30,.34):
        torch.manual_seed(42)
        cfg=make_huananhu_env_cfg(play=True,recover=False)
        cfg.scene.num_envs=64;cfg.seed=42;cfg.auto_reset=False
        wrapped=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg,device='cuda:0'),clip_actions=None)
        env=wrapped.unwrapped
        try:
            state=env.command_manager.get_term('standup');original=state._update_command
            def update(env_ids):
                original(env_ids)
                ids=slice(None) if env_ids is None else env_ids
                state._command[ids]=torch.tensor([0.,0.,command],device=env.device)
            state._update_command=update
            obs,_=wrapped.reset()
            initial_ground_count=int(state.ground_start.sum())
            live=torch.ones(64,dtype=torch.bool,device=env.device);fell=torch.zeros_like(live)
            heights=[];within=[]
            with torch.no_grad():
                for step in range(1000):
                    obs,_,done,_=wrapped.step(actor(obs));done=done.bool()
                    if (step+1)*env.step_dt>5.:
                        mask=live & ~done
                        height=env.scene['robot'].data.root_link_pos_w[mask,2]-env.scene.env_origins[mask,2]
                        heights.append(height);within.append((height-command).abs()<=.03)
                    fell |= live & done & env.reset_terminated
                    live &= ~done
                    if done.any():
                        env.reset(env_ids=done.nonzero().flatten());obs=wrapped.get_observations()
            row=dict(command_height_m=command,fell=int(fell.sum()),mean_height_m=float(torch.cat(heights).mean()),
                     within3cm_fraction=float(torch.cat(within).float().mean()),initial_ground_count=initial_ground_count)
            report['records'].append(row);write(OUT/'source-replay.json',report)
            print('SOURCE_REPLAY '+json.dumps(row),flush=True)
        finally:wrapped.close()
    report['ended']=now();report['exit_code']=0;write(OUT/'source-replay.json',report)


def monitor_physical_failure(env):
    """Observe original failure while allowing continuous, unreplaced rollouts.

    Returning False only suppresses the manager's requirement to reset a fallen
    world before stepping. The original predicate and its state update still run
    once per control step; reward values have no influence on the replay actor.
    """
    from ct_mjlab.reference_training import huananhu_mdp
    env._v11_rollout_failure=huananhu_mdp.failure(env).clone()
    return torch.zeros_like(env._v11_rollout_failure)


def robustness():
    nominal=json.loads((OUT/'nominal/evaluation.json').read_text())
    assert nominal['exit_code']==0
    trigger=max(v['sustained_last5s']['height_tracking'] for v in nominal['poses'].values())>=32
    if not trigger:
        write(OUT/'robustness.json',dict(skipped=True,reason='0.30 m tracking <32/64 in both poses'));return
    actor=actor_from_checkpoint(OUT/'models/model_999.pt')
    report=dict(started=now(),learning=False,num_envs=256,seconds=20.,seed=42,
        trigger='>=32/64 nominal height tracking in at least one pose',records=[],exit_code=None,
        randomization='original training startup/reset/interval events, motor/spring parameters, observation noise and sensor/action delays',
        fall_definition='first original physical/numerical failure; original predicate monitored, physical termination does not request a reset; no reset, no replacement',
        metric_denominator='all 256 environments, including fallen, over all20s and last5s')
    try:
        for command in (.27,.30,.34):
            torch.manual_seed(42)
            cfg=make_replay_cfg(256,play=False)
            cfg.commands['standup'].height_range=(command,command)
            cfg.terminations['failure']=TerminationTermCfg(func=monitor_physical_failure)
            wrapped=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg,device='cuda:0'),clip_actions=None)
            env=wrapped.unwrapped
            try:
                state=env.command_manager.get_term('standup');state.training_iteration=4000
                first=torch.full((256,),-1,dtype=torch.long,device=env.device)
                numeric=torch.zeros(256,dtype=torch.bool,device=env.device)
                ground=state.ground_start.clone()
                hs=torch.zeros(256,device=env.device);tail=hs.clone();within=hs.clone();tail_within=hs.clone()
                obs=wrapped.get_observations()
                with torch.no_grad():
                    for step in range(1000):
                        obs,_,_,_=wrapped.step(torch.where(~numeric[:,None],actor(obs),0.))
                        assert (state.command[:,:2]==0.).all()
                        failure=env._v11_rollout_failure|state.numerical_failure
                        first[(first<0)&failure]=step+1;numeric |= state.numerical_failure
                        height=env.scene['robot'].data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
                        finite=torch.nan_to_num(height,nan=0.,posinf=0.,neginf=0.)
                        hs+=finite;good=((height-command).abs()<=.03)&~numeric;within+=good
                        if step>=750:tail+=finite;tail_within+=good
                fell=first>0
                row=dict(command_height_m=command,initial_ground_count=int(ground.sum()),falls_total=int(fell.sum()),
                    falls_before5s=int((fell&(first<250)).sum()),falls_at_or_after5s=int((first>=250).sum()),
                    numerical_failures=int(numeric.sum()),first_fall_time_s=(first.float()*.02).cpu().tolist(),
                    mean_height_all20s_m=float(hs.mean()/1000),mean_height_last5s_m=float(tail.mean()/250),
                    within3cm_time_fraction_all20s=float(within.mean()/1000),within3cm_time_fraction_last5s=float(tail_within.mean()/250),
                    fallen_samples=fell.nonzero().flatten().cpu().tolist())
                report['records'].append(row);write(OUT/'robustness.json',report)
                print('V11_ROBUSTNESS '+json.dumps(row),flush=True)
            finally:wrapped.close()
            torch.cuda.empty_cache()
        report['exit_code']=0
    except Exception:
        report['exit_code']=1;raise
    finally:report['ended']=now();write(OUT/'robustness.json',report)


if __name__=='__main__':
    torch.set_num_threads(1)
    if len(sys.argv)==2 and sys.argv[1]=='gate':gate()
    elif len(sys.argv)==2 and sys.argv[1]=='source':source_replay()
    elif len(sys.argv)==2 and sys.argv[1]=='robustness':robustness()
    else:main()



