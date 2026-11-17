"""Inspect a frozen high-yaw training stage while the original bounded run continues."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts.agent.fudan_ppo_walk import inspect_randomization_4000 as frozen

ROOT=frozen.ROOT
RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261116_high_yaw10_resume4000_to5000_01'
EXPERIMENT=ROOT/'artifacts/fudan_ppo_walk/20261116-high-yaw10-resume4000-5000-01'
OLD4000=ROOT/'artifacts/fudan_ppo_walk/20261115-random4000-effect-01/nominal4000'
OLD3000=frozen.BASELINE
STAGE4400=ROOT/'artifacts/fudan_ppo_walk/20261116-high-yaw4400-effect-01'


def audit(out,iteration):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    model=RUN/f'model_{iteration}.pt';metadata=torch.load(model,map_location='cpu',weights_only=False)
    assert metadata['iterations']==iteration and not metadata['partial_update']
    assert metadata['primary_updates']==metadata['encoder_updates']==iteration*20
    assert metadata['transitions']==iteration*4096*48+184320
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in metadata[key]['state'].values()}=={iteration*20}
        assert all(torch.isfinite(v).all() for state in metadata[key]['state'].values()
                   for v in state.values() if isinstance(v,torch.Tensor))
    assert all(torch.isfinite(v).all() for v in metadata['model_state_dict'].values())
    restored=json.loads((EXPERIMENT/'initialization-audit.json').read_text());assert restored['passed']
    assert restored['source_checkpoint_sha256']==metadata['resume_from']['sha256']
    assert metadata['command_sampling']['yaw_range']==[-10.,10.] and metadata['command_sampling']['zero_vx_above_yaw']==2.
    events=EventAccumulator(str(RUN),size_guidance={'scalars':0}).Reload()
    assert [v.step for v in events.Scalars('Train/reward') if v.step<=iteration]==list(range(4001,iteration+1))
    finished=iteration==5000;final_sha=None
    if finished:
        from scripts.agent.fudan_ppo_walk.resume_pd200_random_from3000 import same
        report=json.loads((RUN/'training_report.json').read_text())
        assert report['stopped_by']=='iteration_budget' and report['iterations_added']==report['iteration_budget']==1000
        assert json.loads((EXPERIMENT/'training-audit.json').read_text())['budget_and_learning_state_passed']
        final=torch.load(RUN/'model_final.pt',map_location='cpu',weights_only=False)
        for key in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng',
                    'iterations','transitions','primary_updates','encoder_updates','command_sampling'):
            assert same(final[key],metadata[key]),key
        final_sha=frozen.sha(RUN/'model_final.pt')
    checks={};drift=[]
    for name,digest in json.loads((RUN/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            checks[name]=frozen.sha(RUN/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest
            assert checks[name],name
            if frozen.sha(ROOT/name)!=digest:drift.append(name)
        elif name.startswith('resources/'):
            checks[name]=frozen.sha(ROOT/name)==digest;assert checks[name],name
    out.mkdir(parents=True,exist_ok=False);(out/'checkpoint').mkdir();shutil.copy2(model,out/'checkpoint'/model.name)
    frozen.write(out/'checkpoint-audit.json',dict(passed=True,iteration=iteration,checkpoint_sha256=frozen.sha(model),
        source_checkpoint=str(model),training_finished=finished,final_checkpoint_sha256=final_sha,original_training_not_interrupted=True,
        snapshot_and_resource_checks=checks,current_workspace_drift=drift,inherited_abandoned_rollout_transitions=184320,
        command_sampling=metadata['command_sampling'],optimizer_updates=0,behavior_accepted=False))
    shutil.copy2(__file__,out/'inspection-source.py')


def high_yaw(out,iteration,attitude=False):
    import torch
    import numpy as np
    from scripts.agent.fudan_ppo_walk import evaluate
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record
    torch.set_num_threads(1)
    conditions=replace(from_record(json.loads((EXPERIMENT/'budget.json').read_text())['disturbances']),
        pushes=False,external_force_range=(0.,0.),external_torque_range=(0.,0.))
    cases=[]
    for height in (.22,.26,.30):
        for yaw in (-10.,-5.,-3.,3.,5.,10.):
            case=dict(id=f'high_yaw_{yaw:g}_{height:g}',category='spin',duration=7.,
                segments=[dict(start=0.,duration=7.,command=[0.,yaw,height])])
            cases.append(dict(id=case['id']+'__50001',case=case,seed=50001,randomized=False))
    original=evaluate.configure_samples
    paths=[('3000',ROOT/'artifacts/fudan_ppo_walk/models/20261114_model3000_pd200_p20d1_half_random_delay10to20.pt'),
           ('4000',ROOT/'artifacts/fudan_ppo_walk/models/20261115_model4000_pd200_p20d1_full_random_ground04to12_force3_push.pt'),
           (str(iteration),out/'checkpoint'/f'model_{iteration}.pt')]
    if attitude:paths=paths[-1:]
    for label,model in paths:
        target=out/('attitude-diagnostic' if attitude else 'high-yaw')/label
        proof=dict(integrations=0,pd_updates=0,PD_formula_max_error=0.,external_wrench_peak=0.);attitude_frames=[]
        if target.exists():
            # 后续阶段只复用已验证且条件完全相同的旧模型记录，不重复仿真。
            assert label in ('3000','4000') and target.is_symlink()
            assert target.resolve()==STAGE4400/'high-yaw'/label
            protocol=json.loads((target/'protocol.json').read_text())
            assert protocol['samples']==cases and protocol['disturbances']==conditions.record()
            assert json.loads((target/'runtime-audit.json').read_text())['passed']
            assert json.loads((target/'execution.json').read_text())['complete_traces']==18
            continue
        def configured(env,batch,folder):
            value=original(env,batch,folder);a=env.action;sim=env.env.sim;apply=a.apply_actions
            assert (a.cfg.leg_stiffness,a.cfg.leg_damping,a.cfg.wheel_stiffness,a.cfg.wheel_damping)==(20.,1.,.1,0.)
            assert a.cfg.pd_decimation==5 and sim.mj_model.opt.timestep==.001 and a.motor_strength.eq(1).all()
            def observed():
                if attitude:
                    attitude_frames.append(torch.cat((sim.data.qpos[:,3:7],sim.data.qvel[:,3:6]),-1).clone())
                update=a._pd_tick==0;apply();proof['integrations']+=1
                if update:
                    q=sim.data.qpos[:,a._motor_q_adr];v=sim.data.qvel[:,a._motor_v_adr];legs=a._leg_action_ids;wheels=a._wheel_action_ids
                    pos=a._nominal_pos+.5*a._control_actions
                    pos[:,legs]=torch.maximum(torch.minimum(pos[:,legs],a._reference_pos[:,legs]+3.14),a._reference_pos[:,legs]-3.14)
                    expected=torch.zeros_like(q);expected[:,legs]=20*(pos[:,legs]-q[:,legs])-v[:,legs]
                    expected[:,wheels]=.1*((10*a._control_actions[:,wheels]).clamp(-150,150)-v[:,wheels])
                    expected=expected.clamp(-a._effort_limits,a._effort_limits)
                    proof['PD_formula_max_error']=max(proof['PD_formula_max_error'],float((expected-a.torques).abs().max()))
                    proof['external_wrench_peak']=max(proof['external_wrench_peak'],float(sim.data.xfrc_applied.abs().max()))
                    proof['pd_updates']+=1
            a.apply_actions=observed;return value
        request=SimpleNamespace(checkpoint=model,output=target,deadline=time.time()+600,diagnostic=True,
            diagnostic_height=None,batch_size=18,selection=None,leg_load_profile='legacy',response_parameters=None)
        with patch.object(evaluate,'evaluation_items',return_value=cases),patch.object(evaluate,'from_sim2sim_config',return_value=conditions),patch.object(evaluate,'configure_samples',configured):
            evaluate.gpu(request)
        assert proof['integrations']==7000 and proof['pd_updates']==1400 and proof['PD_formula_max_error']<2e-5 and proof['external_wrench_peak']==0.
        frozen.write(target/'runtime-audit.json',dict(passed=True,**proof,optimizer_updates=0))
        if attitude:
            values=torch.stack(attitude_frames).cpu().numpy();t=np.arange(len(values))*.001;last=(t>=6)&(t<7)
            np.savez_compressed(target/'attitude-1ms.npz',time=t,quaternion=values[:,:,:4],angular_body=values[:,:,4:])
            q=values[:,:,:4];w,x,y,z=np.moveaxis(q,-1,0)
            gravity=np.stack((2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1),axis=-1)
            omega=values[:,:,4:];perpendicular=np.linalg.norm(np.cross(omega,gravity),axis=-1)
            derivative=np.linalg.norm(np.diff(gravity,axis=0)/.001,axis=-1);rows=[]
            for i,sample in enumerate(cases):
                with np.load(target/'traces'/(sample['id']+'.npz')) as trace:
                    alignment=float(np.max(np.abs(trace['angular_body'][:-1]-omega[:,i])))
                    assert alignment<1e-6
                rows.append(dict(sample_id=sample['id'],body_xy_last1_peak_rad_s=float(np.linalg.norm(omega[last,i,:2],axis=-1).max()),
                    gravity_perpendicular_last1_peak_rad_s=float(perpendicular[last,i].max()),
                    gravity_derivative_last1_peak_rad_s=float(derivative[(t[1:]>=6)&(t[1:]<7),i].max()),
                    quaternion_omega_alignment_max_error=alignment))
            frozen.write(target/'attitude-diagnostic.json',dict(rows=rows,physics_dt=.001,optimizer_updates=0,
                scope='Diagnostic decomposition along gravity; does not replace the declared body-XY acceptance threshold or alter pass/fail'))


def report(out,iteration):
    import numpy as np
    from scripts.agent.fudan_ppo_walk import inspect_pd200_training as inspect
    inspect.RUN=RUN;inspect.report(out/'nominal')
    current=json.loads((out/'nominal/summary.json').read_text());before=json.loads((OLD4000/'summary.json').read_text())
    original=json.loads((OLD3000/'summary.json').read_text())
    groups={};rows=[]
    for label in ('3000','4000',str(iteration)):
        folder=out/'high-yaw'/label
        if not (folder/'acceptance/acceptance.json').exists():continue
        accepted=json.loads((folder/'acceptance/acceptance.json').read_text());scored={r['sample_id']:r for r in accepted['samples']}
        protocol=json.loads((folder/'protocol.json').read_text());model_rows=[]
        for sample in protocol['samples']:
            with np.load(folder/'traces'/(sample['id']+'.npz')) as trace:
                mask=(trace['time']>=2)&(trace['time']<7);cmd=sample['case']['segments'][0]['command']
                model_rows.append(dict(sample_id=sample['id'],model=label,yaw_command=cmd[1],height_command=cmd[2],
                    yaw_mean_rad_s=float(trace['angular_body'][mask,2].mean()),
                    yaw_mae_rad_s=float(np.abs(trace['angular_body'][mask,2]-cmd[1]).mean()),
                    vx_abs_mean_m_s=float(np.abs(trace['linear_body'][mask,0]).mean()),
                    world_horizontal_speed_peak_m_s=float(np.linalg.norm(trace['linear_world'][mask,:2],axis=1).max()),
                    pitch_peak_deg=float(abs(trace['roll_pitch_deg'][mask,1]).max()),
                    wheel_support_mean_N=trace['wheel_support'][mask].mean(0).tolist(),
                    body_contact_peak_N=float(trace['base_contact'][mask].max()),passed=scored[sample['id']]['passed'],
                    complete=scored[sample['id']]['complete'],finite=scored[sample['id']]['finite']))
        groups[label]=dict(passed=accepted['technical_passed'],total=accepted['total'],rows=model_rows);rows.extend(model_rows)
    zero=[];plans=json.loads((out/'nominal/plan.json').read_text())['samples']
    for height in (.22,.26,.30):
        idx=next(i for i,s in enumerate(plans) if s['case']['category']=='zero' and s['case']['segments'][0]['command'][2]==height)
        sid=plans[idx]['id'];values=[]
        for label,folder,summary in [('3000',OLD3000,original),('4000',OLD4000,before),(str(iteration),out/'nominal',current)]:
            with np.load(folder/'motor-pd200.npz') as motor:
                mask=(motor['time']>=29)&(motor['time']<30)
                pp=float(np.ptp(np.rad2deg(motor['values'][mask,idx][:,[6,7,9,10]]),axis=0).max())
            row=next(r for r in summary['rows'] if r['sample_id']==sid)
            values.append(dict(model=label,leg_last1_pp_deg_200Hz=pp,total_displacement_m=row['total_displacement_m'],passed=row['passed']))
        zero.append(dict(height=height,models=values))
    finished=json.loads((out/'checkpoint-audit.json').read_text())['training_finished']
    frozen.write(out/'comparison.json',dict(iteration=iteration,is_final=finished,nominal_passed={
        '3000':original['technical_passed'],'4000':before['technical_passed'],str(iteration):current['technical_passed']},
        zero=zero,high_yaw=groups,reward=current['reward'],optimizer_updates=0,training_original_budget_not_changed=True))
    if groups:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for row,sign in enumerate((1,-1)):
            for col,magnitude in enumerate((3,5,10)):
                target=sign*magnitude;sid=f'high_yaw_{target:g}_0.26__50001'
                ax=axes[row,col];ax.axhline(target,color='black',ls='--',lw=1,label='Command')
                for label in groups:
                    with np.load(out/'high-yaw'/label/'traces'/(sid+'.npz')) as trace:
                        ax.plot(trace['time'],trace['angular_body'][:,2],lw=.8,label=label)
                ax.set(title=f'h=0.26m, yaw target {target:+g} rad/s',xlabel='Time (s)',ylabel='Actual yaw rate (rad/s)')
                ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(out/'high-yaw-tracking.png',dpi=150);plt.close(fig)


def video_inputs(out,iteration):
    samples=['zero_0_0_0.26__50001','high_yaw_10_0.26__50001','high_yaw_-10_0.26__50001']
    previous=STAGE4400 if iteration!=5000 else ROOT/'artifacts/fudan_ppo_walk/20261116-high-yaw4800-effect-01'
    previous_label='4400' if iteration!=5000 else '4800'
    before=('4000',OLD4000,out/'high-yaw/4000') if iteration==4400 else (previous_label,previous/'nominal',previous/'high-yaw'/previous_label)
    models=[('before',*before),('after',str(iteration),out/'nominal',out/'high-yaw'/str(iteration))]
    for side,label,nominal,high in models:
        directory=out/'paired-video-inputs'/side;folder=directory/'no_external'
        (folder/'fixtures').mkdir(parents=True);(folder/'traces').mkdir()
        (folder/'fixtures/model.mjb').symlink_to(nominal/'gpu/fixtures/model.mjb')
        chosen=[]
        for sid in samples:
            source=nominal/'gpu' if sid.startswith('zero') else high
            protocol=json.loads((source/'protocol.json').read_text());chosen.append(next(s for s in protocol['samples'] if s['id']==sid))
            (folder/'traces'/(sid+'.npz')).symlink_to(source/'traces'/(sid+'.npz'))
        frozen.write(folder/'protocol.json',dict(samples=chosen,scope='Rendering links to verified GPU trajectories; no physics'))
        checkpoint=Path(json.loads((nominal/'plan.json').read_text())['checkpoint'])
        frozen.write(directory/'run.json',dict(checkpoint=str(checkpoint),sha256=frozen.sha(checkpoint),
            display_label=f'Iteration {label}',leg_p=20.,leg_d=1.))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('audit','nominal','high-yaw','attitude','report','video-inputs','videos'))
    p.add_argument('--iteration',type=int,default=4400);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();iteration=args.iteration
    if args.mode=='audit':audit(out,iteration);return
    if args.mode=='video-inputs':video_inputs(out,iteration);return
    if args.mode=='videos':frozen.videos(out);return
    proof=json.loads((out/'checkpoint-audit.json').read_text());assert proof['iteration']==iteration and proof['passed']
    assert frozen.sha(out/'checkpoint'/f'model_{iteration}.pt')==proof['checkpoint_sha256']
    frozen.RUN=RUN;sys.meta_path.insert(0,frozen.FrozenTrainingSource())
    if args.mode in ('high-yaw','attitude'):high_yaw(out,iteration,attitude=args.mode=='attitude')
    elif args.mode=='report':report(out,iteration)
    else:
        from scripts.agent.fudan_ppo_walk import inspect_pd200_training as inspect
        from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record
        import torch
        model=out/'checkpoint'/f'model_{iteration}.pt';metadata=torch.load(model,map_location='cpu',weights_only=False)
        conditions=replace(from_record(metadata['disturbances']),pushes=False,external_force_range=(0.,0.),external_torque_range=(0.,0.))
        inspect.MODEL=model;inspect.RUN=RUN;inspect.EXPERIMENT=EXPERIMENT
        inspect.run(out/'nominal',checkpoint_audit=lambda _:proof['snapshot_and_resource_checks'],conditions_override=conditions)
        plan=json.loads((out/'nominal/plan.json').read_text());plan['is_final_checkpoint']=proof['training_finished']
        frozen.write(out/'nominal/plan.json',plan)
    frozen.write(out/(args.mode+'-frozen-execution.json'),dict(modules=frozen.LOADED,optimizer_updates=0))


if __name__=='__main__':main()
