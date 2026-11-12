"""Check split posture penalties, unchanged inputs and legacy metadata; no training."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ct_mjlab.envs.fudan_ppo_walk import posture_reward
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT, RESET_V2_CONTRACT
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner
from ct_mjlab.scripts import train
from .check_resume import equal
from .reset_v2_fixture import fixture


def gravity(roll, pitch, device='cpu'):
    r = torch.as_tensor(roll, device=device, dtype=torch.float32) * math.pi / 180.
    p = torch.as_tensor(pitch, device=device, dtype=torch.float32) * math.pi / 180.
    return torch.stack((p.sin(), -r.sin()*p.cos(), -r.cos()*p.cos()), -1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    checks = []
    def verify(name, value):
        assert bool(value), name
        checks.append(name)
    def forbidden(*a, **kw):
        raise AssertionError('No actual physics, neural-network forward or optimizer update')
    profile = posture_reward.from_task_config()
    verify('default_forward_band0to5_and_ratio1over20',profile['pitch_range_deg']==[0.,5.] and profile['pitch_inner_ratio']==.05)
    verify('default_separate_roll_pitch17_rewards',len(FudanWalkCfg().rewards.scales)==17 and 'orientation' not in FudanWalkCfg().rewards.scales)
    verify('old_checkpoint_uses_legacy16_rewards',posture_reward.from_record(None)==posture_reward.LEGACY and len(posture_reward.reward_names(None))==16)
    for change in ({'pitch_inner_ratio':-1.},{'pitch_range_deg':[5.,0.]},{'roll_scale':float('nan')},{'version':'unknown'}):
        try:posture_reward.from_record(dict(profile,**change))
        except ValueError: rejected=True
        else: rejected=False
        verify('reject_'+next(iter(change)),rejected)

    with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2',
                     'mjlab.envs.ManagerBasedRlEnv.step','torch.nn.Linear.forward','torch.optim.Adam.step'):
            stack.enter_context(patch(name,side_effect=forbidden))
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            g=gravity([0.,0.,0.,0.,3.,3.],[-2.,0.,2.,6.,0.,2.],device)
            env=SimpleNamespace(gravity=g,posture_reward_profile=profile);before=g.clone()
            values=posture_reward.penalties(env);two=math.radians(2.)**2
            verify(device+'_backward_pitch_keeps_full_strength',torch.isclose(values['pitch'][0],g.new_tensor(two),atol=1e-8))
            verify(device+'_forward2deg_uses_one_twentieth',torch.isclose(values['pitch'][2],g.new_tensor(two*.05),atol=1e-8))
            verify(device+'_zero_is_not_forced_to2deg',values['pitch'][1]==0 and values['pitch'][2]>0)
            verify(device+'_roll_is_independent_of_pitch',torch.isclose(values['roll'][4],values['roll'][5],atol=1e-8) and values['roll'][4]>0)
            verify(device+'_above5deg_increases_penalty',values['pitch'][3]>values['pitch'][2])
            boundary=SimpleNamespace(gravity=gravity([0.,0.],[4.9999,5.0001],device),posture_reward_profile=profile)
            edge=posture_reward.penalties(boundary)['pitch']
            verify(device+'_continuous_at_upper_boundary',abs(edge[1]-edge[0])<1e-7)
            verify(device+'_measurements_not_modified',torch.equal(g,before))
            env.posture_reward_profile=posture_reward.LEGACY
            verify(device+'_legacy_exact_original_gravity_penalty',torch.equal(posture_reward.penalties(env)['orientation'],g[:,:2].square().sum(-1)))

        # Exercise the real reward aggregation with synthetic geometry, not a copy of it.
        env=fixture(4);env.randomized=False;env.gravity=gravity([0.,0.,3.,0.],[2.,-2.,2.,6.])
        model=env.env.sim.mj_model.model
        env.wheel_ids=torch.tensor([model.body(x).id for x in ('lfwheel','rfwheel')])
        env.hip_ids=torch.tensor([model.joint(x).id for x in ('lf0','rf0')])
        env.front_q=torch.tensor([int(model.joint(x).qposadr[0]) for x in ('lf0','lf1','rf0','rf1')])
        env.front_limits=torch.tensor([[-100.,100.]]*4)
        env.env.sim.data.xpos=torch.zeros(4,model.nbody,3)
        env.env.sim.data.xanchor=torch.zeros(4,model.njnt,3)
        _,old_terms=FudanEnv.reward(env)
        posture_reward.configure(env.cfg,profile);env.posture_reward_profile=profile
        total,new_terms=FudanEnv.reward(env)
        verify('actual_reward_has17_terms_and_no_double_orientation',len(new_terms)==17 and 'orientation' not in new_terms)
        verify('other15_reward_contributions_unchanged',all(torch.equal(value,new_terms[name]) for name,value in old_terms.items() if name!='orientation'))
        verify('total_equals_sum_of_logged_terms',torch.equal(total,torch.stack(list(new_terms.values())).sum(0)))
        verify('independent_roll_pitch_clipping',new_terms['roll'].ge(-env.dt).all() and new_terms['pitch'].ge(-env.dt).all())
        for contract in (RESET_V2_CONTRACT,RESET_CONTRACT):
            env=fixture(4,contract);env.randomized=False
            before=env.observations();history=env._history_frame.clone()
            posture_reward.configure(env.cfg,profile);env.posture_reward_profile=profile
            after=env.observations()
            verify(contract+'_actor_critic_history_unchanged',all(torch.equal(a,b) for a,b in zip(before,after)) and torch.equal(history,env._history_frame))

        parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        env=fixture(4,parent['reset_contract']);posture_reward.configure(env.cfg,profile);env.posture_reward_profile=profile
        env.command_sampling=command_config();env.disturbance_config=from_record(parent.get('disturbances'));env.action.leg_load=None
        with redirect_stdout(io.StringIO()):
            learner=FudanRunner(env,log_dir=Path(temporary)/'learner',device='cpu')
            restored=learner.load(args.checkpoint);saved=learner.save('metadata-check-only.pt')
        metadata=torch.load(saved,map_location='cpu',weights_only=False)
        verify('resume_records_legacy_source_and_split_target',restored['source_posture_reward']==posture_reward.LEGACY and restored['posture_reward']==profile)
        verify('new_checkpoint_records_profile_and17_reward_names',metadata['posture_reward']==profile and metadata['reward_names']==list(env.cfg.rewards.scales))
        for key in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng'):equal(metadata[key],parent[key])
        verify('metadata_save_does_not_train',metadata['iterations']==parent['iterations']);learner.writer.close()

        proof=dict(passed=True,reset_contract=parent['reset_contract'],kind='physical-visible-precheck',num_envs=20,
                   physics_steps=1,disturbances=env.disturbance_config.record(),command_sampling=command_config())
        path=Path(temporary)/'synthetic-proof.json';path.write_text(json.dumps(proof))
        argv=['--task','fudan_ppo_walk','--num-envs','4','--iterations','1','--resume',str(args.checkpoint),
              '--preserve-history-v2','--precheck',str(path),'--log-dir',str(Path(temporary)/'never-train')]
        with patch.object(train.task_registry,'make_env',side_effect=forbidden):
            try:train.main(argv)
            except ValueError as exc:verify('old_reward_precheck_rejected','posture reward' in str(exc))
            else:raise AssertionError('Stale precheck approved split rewards')
        class GatePassed(Exception):pass
        proof['posture_reward']=profile;path.write_text(json.dumps(proof))
        with patch.object(train.task_registry,'make_env',side_effect=GatePassed):
            try:train.main(argv)
            except GatePassed:verify('matching_profile_precheck_accepted',True)
            else:raise AssertionError('Expected mocked environment gate')

    result=dict(passed=True,count=len(checks),checks=checks,posture_reward=profile,
                physics_steps=0,model_forwards=0,optimizer_updates=0,
                scope='CPU/CUDA posture math, actual reward/input/metadata/CLI with nonphysical fixtures; no behavioral validation')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
