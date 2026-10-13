"""No-learning finite, reward-isolation, curriculum and full-migration checks."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo
sys.dont_write_bytecode = True
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import height_curriculum as h


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def course_checks():
    c = h.HeightCurriculum()
    c.learning_active = True
    c.completed_updates = 2549
    for step in range(120):
        assert c.advance(1., .02, step) is None
    assert c.level == .21 and c.hold_s >= 2.
    c.completed_updates = 2550
    assert c.advance(1.,.02,120)['level_m'] == .22
    # Dwell and the50-update gate are independent simultaneous conditions.
    c = h.HeightCurriculum()
    c.learning_active = True
    c.completed_updates = 2550
    for step in range(120,219):
        assert c.advance(2/3, .02, step) is None
    assert c.level == .21
    snapshot = c.state_dict()
    restored = h.HeightCurriculum()
    restored.load(snapshot)
    assert restored.state_dict() == snapshot
    restored.learning_active = True
    assert restored.advance(2/3,.02,219)['level_m'] == .22
    assert restored.level_start_updates == 2550
    for step in range(220,350):
        assert restored.advance(1.,.02,step) is None
    restored.completed_updates = 2600
    restored.advance(.5,.02,350)
    for step in range(351,400):
        restored.advance(1.,.02,step)
    restored.advance(.5,.02,400)
    assert restored.hold_s == 0. and restored.level == .22
    for step in range(401,501):
        restored.advance(1.,.02,step)
    assert restored.level == .23
    assert h.HeightCurriculum().height_range == (.20,.21)
    restored.level_index = 13
    assert restored.height_range == (.27,.34)
    return dict(passed=True,minimum50_actual_updates=True,continuous2s=True,
                below_two_thirds_resets_hold=True,state_resume_exact=True,final_range=[.27,.34])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,nargs='+',default=[16,4096])
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if any(n not in (16,4096) for n in args.num_envs):
        p.error('Use required16/4096 populations; local only16')
    args.output.mkdir(parents=True,exist_ok=False)
    original=h.validate_source(args.source)
    report=dict(command=shlex.join([sys.executable,*sys.argv]),started=now(),ended=None,
        learning=False,source_sha256=h.SOURCE_SHA256,device=args.device,
        versions={n:importlib.metadata.version(n) for n in ('torch','mjlab','mujoco','mujoco-warp','rsl-rl-lib')},
        gpu=torch.cuda.get_device_name(args.device),steps=64,cases={},passed=False,
        curriculum_checks=course_checks(),exit_code=None)
    root=Path(__file__).resolve().parents[2]
    frozen={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest()
        for folder in ('mjlab_training/src/ct_mjlab/experiments','mjlab_training/src/ct_mjlab/reference_training')
        for x in (root/folder).glob('*.py') if x.name!='height_curriculum.py'}
    report['frozen_sources_sha256']=frozen
    report['height_module_sha256']=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
    report['check_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        reference=h.serializable(h.source.make_env_cfg('baseline'))
        for case in h.CASES:
            encoded=h.serializable(h.make_env_cfg(case))
            normalized=json.loads(json.dumps(encoded))
            normalized['commands']=reference['commands']
            normalized['rewards']=reference['rewards']
            assert normalized==reference,'Unexpected physical/observation/action/event/termination changes'
            expected_weights={k:v['weight'] for k,v in reference['rewards'].items()}
            for name,term in encoded['rewards'].items():
                assert term['weight']==(-.5 if name=='knee_limit' and case=='curriculum_knee' else expected_weights[name])
            for n in args.num_envs:
                torch.manual_seed(42)
                env=RslRlVecEnvWrapper(ManagerBasedRlEnv(h.make_env_cfg(case,n),device=args.device),clip_actions=None)
                try:
                    raw=env.unwrapped;state=raw.command_manager.get_term('standup')
                    numerical=0;per_term_finite=True
                    for _ in range(64):
                        obs,reward,_,_=env.step(torch.zeros(n,6,device=args.device))
                        assert obs['actor'].shape==(n,35) and obs['critic'].shape==(n,78)
                        assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all()
                        assert all(torch.isfinite(v).all() for v in raw._huananhu_rewards.values.values())
                        # Auto-reset clears the numerical flag; use original per-transition metric too.
                        numerical+=state.last_transition_numerical_failure_count
                        assert (state.command[:,:2]==0).all() and (state.ground_command==0).all()
                        assert ((state.sampled[:,2]>=.20)&(state.sampled[:,2]<=.21)).all()
                        assert (state.special_mode==-1).all() and not state.heading.any()
                    assert numerical==0
                    before=raw.sim.data.qpos.clone(); velocity=raw.sim.data.qvel.clone()
                    original_cmd=state.command.clone()
                    baseline={k:v.clone() for k,v in h.height_base.RewardState(raw).compute(raw).items()}
                    variant=h.RewardState(raw).compute(raw)
                    difference={k:float((variant[k]-baseline[k]).abs().max()) for k in baseline}
                    changed={k for k,v in difference.items() if v!=0.}
                    assert changed<={'track_height_exp_tight'}
                    assert difference['track_height_exp_tight']>0.
                    # Same simulated state, same height; isolate only speed command dependencies.
                    state.command[:,:2]=torch.tensor([.8,.4],device=args.device)
                    moving=h.height_base.RewardState(raw).compute(raw)
                    state.command.copy_(original_cmd)
                    speed_changed={k for k in baseline if not torch.equal(moving[k],baseline[k])}
                    speed_allowed=set(h.height_base.GATED_TERMS)|{'stand_still_lin_vel','flat_orientation_y_v','flat_orientation_x_v'}
                    assert speed_changed<=speed_allowed
                    assert torch.equal(before,raw.sim.data.qpos) and torch.equal(velocity,raw.sim.data.qvel)
                    knee_formula=variant['knee_limit']
                    knee_weight=raw.cfg.rewards['knee_limit'].weight if 'knee_limit' in raw.cfg.rewards else 0.
                    reference_knee=h.huananhu_rewards.RewardState(raw).compute(raw)['knee_limit']
                    actual_knee=knee_weight*knee_formula
                    assert torch.equal(knee_formula,reference_knee)
                    assert torch.equal(actual_knee,(-.5 if case=='curriculum_knee' else 0.)*reference_knee)
                    entry=dict(num_envs=n,steps=64,finite=True,all_reward_terms_finite=per_term_finite,
                        numerical_failures=numerical,zero_velocity_special_heading_ground_override=True,
                        configuration_isolation=True,reward_changed=sorted(changed),reward_max_abs_delta=difference,
                        speed_command_changed=sorted(speed_changed),knee_weight=knee_weight,
                        knee_formula_original_exact=True,same_qpos_qvel_for_reward_isolation=True)
                    if n==4096:
                        cfg=h.height_base.runner_cfg(42)
                        runner=h.HeightRunner(env,cfg,None,case=case,device=args.device)
                        runner.load(args.source,migration=True)
                        path=args.output/case/'model_resume_initial.pt';path.parent.mkdir(exist_ok=False)
                        runner.save(str(path));saved=torch.load(path,map_location='cpu',weights_only=False)
                        equal={k:h.source.state_equal(original[k],saved[k]) for k in
                            ('actor_state_dict','critic_state_dict','optimizer_state_dict')}
                        assert all(equal.values()) and saved['infos']['completed_updates']==2500
                        assert saved['infos']['height_curriculum']['level_m']==.21
                        h.validate_checkpoint(saved,case)
                        runner.load(path)
                        assert state.height_curriculum.state_dict()==saved['infos']['height_curriculum']
                        entry.update(complete_state_equal=equal,initial_updates=2500,initial_level_m=.21,
                            initial_learning_rate=runner.alg.learning_rate,own_checkpoint_reload=True,
                            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        del runner,saved
                    report['cases'][f'{case}:{n}']=entry
                    print('HEIGHT_PREFLIGHT '+json.dumps(entry),flush=True)
                    (args.output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
                finally:
                    env.close()
                del env
                torch.cuda.empty_cache()
        report['passed']=True;report['exit_code']=0
    except Exception:
        report['exit_code']=1
        raise
    finally:
        report['ended']=now()
        report['frozen_sources_unchanged']=all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in frozen.items())
        (args.output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
