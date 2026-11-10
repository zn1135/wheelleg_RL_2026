#!/usr/bin/env python3
"""Current-reward V12 retraining versus three sealed historical motion models."""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from mjlab.utils.lab_api.math import quat_mul
from scripts.agent.supermimic import score_supermimic_physical_prediction as core
from ct_mjlab.supermimic.motion import WorldModel, physical47_from_full64
from ct_mjlab.supermimic.networks import RunningNorm
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, WORLD_LOSS_RECIPE
from ct_mjlab.supermimic.state import NOISE, rotation_vector, unit_quat

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/supermimic/supermimic-v12-retrain-20261031-r1'
V12='chuanliantui-supermimic-motion-only-pd-reward-soft-termination-v12'
OLD=ROOT/'artifacts/supermimic/supermimic-v10-world-only-20261031-r1'
OLD_RUN=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v10_contact_worldonly_r1'
V10_PT=ROOT/'artifacts/supermimic/supermimic-v10-one-step-vs-rollout-20261031-r3/result/predictions.pt'
V11_PT=ROOT/'artifacts/supermimic/supermimic-v11-world-only-20261031-r1/score/predictions.pt'
PREVIOUS_PT=ROOT/'artifacts/supermimic/supermimic-v12-motion-only-20261031-r2/score/predictions.pt'
PARENT=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v7_worldfit4_actortransfer3600/checkpoint_000615.pt'
REMOVED_REWARD_TERMS={'lin_vel_z','base_link_contact','wheels_airborne','action_rate','action_smooth'}
LOCKED_NEW={
    'supermimic/__main__.py':'bea771fd2e0686b53a196c78f5d488df95471ae1878a56b6f6d01502ddcb809c',
    'supermimic/learner.py':'69f2c43ea581d70c2acc6999265dbb1f214143e25283f7bfe5ddaccb04cda635',
    'supermimic/motion.py':'1ce8e3a6332f284b21d3db5f181fd502baa3359bc8048aad0b25a0c9f5e449b9',
    'supermimic/motion_adapter.py':'7b85cf54eef2268af61ce67911fd68b2bd8268b7bd02e1dea8ea76bbf652e1a0',
    'supermimic/analytic_reward.py':'eaeb060864a7b9637cbf9e287a6651df0560a7d513abfd62b80715e2ee3cf327',
    'envs/chuanliantui_split/rewards.py':'0128333915b48dd9e808e9e96b864fb034a8ca5beab6e8a40176fa89804a3757'}
METRICS=('root_linear_velocity_m_s','root_angular_velocity_rad_s','LINK_vx_m_s','orientation_geodesic_degrees',
    'nonwheel_hinge_q_rad','slider_q_m','hinge_velocity_including_wheels_rad_s','slider_velocity_m_s')


def sha(path):return core.digest(path)


def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.cpu().contiguous().numpy().tobytes()==b.cpu().contiguous().numpy().tobytes()


def source_bridge():
    source=ROOT/'mjlab_training/src/ct_mjlab';current={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    old=json.loads((OLD_RUN/'source_sha256.json').read_text())
    assert sha(OLD_RUN/'source_sha256.json')=='f9297b547d969d5226e81de5d8cd678faf28f2ea8bc4fc99790f356852609c86'
    assert len(old)==99 and len(current)==101 and set(current)-set(old)=={'supermimic/motion.py','supermimic/motion_adapter.py'}
    for name,value in old.items():
        assert sha(OLD_RUN/'snapshot/ct_mjlab'/name)==value
        assert current[name]==(LOCKED_NEW[name] if name in LOCKED_NEW else value),name
    assert all(current[name]==value for name,value in LOCKED_NEW.items())
    def function(path,name):
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.dump(node,include_attributes=False)
    old_main=OLD_RUN/'snapshot/ct_mjlab/supermimic/__main__.py';new_main=source/'supermimic/__main__.py'
    assert function(old_main,'collect')==function(new_main,'collect')
    return current,old,dict(world_source_count=101,capture_source_count=99,unchanged_sources=95,
        reviewed_changes_and_additions=LOCKED_NEW,collect_ast_equal=True,
        source_roles='Current v7/13-term task/world; historical v5/18-term recordings remain historical.',
        make_env_difference='Motion PD validation; source hash allowlist separately covers explicit reward-only changes, not physics/actor/control changes.')


def recordings(old,current):
    current_infos=core.contracts(core.command_cfg('walk'));records=[]
    assert sha(PARENT)=='50ee90a69e6f58344898277a77ff850d2504dbda31dd3ff4b8beb6473ef5105f'
    old_infos=torch.load(PARENT,map_location='cpu',weights_only=False,mmap=True)['infos']
    differences={key for key in old_infos.keys()|current_infos.keys() if old_infos.get(key)!=current_infos.get(key)}
    assert differences=={'reward_recipe','reward_weights'}
    assert old_infos['reward_recipe']=='split-walk-relative-velocity-parking-noboost-tilt5-symmetry3cap10-v5'
    assert current_infos['reward_recipe']=='split-walk-relative-velocity-parking-noboost-tilt5-symmetry3cap10-no-vz-contact-rate-smooth-v7'
    assert current_infos['reward_weights']=={key:value for key,value in old_infos['reward_weights'].items() if key not in REMOVED_REWARD_TERMS}
    for name in ('midpoint-actor','final-actor'):
        directory=OLD/name;sources=json.loads((directory/'source-sha256.json').read_text())
        prefix='mjlab_training/src/ct_mjlab/'
        assert {k[len(prefix):]:v for k,v in sources.items() if k.startswith(prefix)}==old
        for relative,value in sources.items():
            path=Path(relative);assert not path.is_absolute() and '..' not in path.parts
            assert sha(directory/'snapshot'/path)==value
            key=relative.removeprefix(prefix)
            assert sha(ROOT/path)==(current[key] if relative.startswith(prefix) else value)
        # Original loader retains its actual historical ROOT check; no digest or label is patched.
        with patch.object(core,'ROOT',directory/'snapshot'):record=core.load_recording(directory,old,old_infos)
        records.append(record)
    return records,current_infos,old_infos


def historical_windows(records):
    path=OLD_RUN/'snapshot/ct_mjlab/supermimic/analytic_reward.py'
    assert sha(path)=='f9372f6c21b0f22948363e85bcbbbe439a9de131f14c02fe93cb0ad3ed03855e'
    spec=importlib.util.spec_from_file_location('_historical_sm_18_term_reward',path)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    assert len(legacy.WEIGHTS)==18
    # These are the actual archived formulas, not a removed assertion or retagged reward.
    with patch.object(core,'differentiable_reward',legacy.differentiable_reward), \
         patch.object(core,'WEIGHTS',legacy.WEIGHTS),patch.object(core,'TERMS',tuple(legacy.WEIGHTS)):
        windows=[core.make_windows(record) for record in records]
    return windows,dict(path=str(path),sha256=sha(path),term_count=18,
        original_total_reward_assertion_retained=True,old_labels_reweighted=False)


def load_references(windows):
    assert sha(V10_PT)=='b0646cd49a6f019b7ddf2cf0735a8b085230be1ff064c858fb25fffac08d3eb2'
    assert sha(V11_PT)=='96a3474db724acf79603b794d447f1daf8cf4e51fd701403a9a1a87abdff021b'
    assert sha(PREVIOUS_PT)=='3834d75c920d83ad85169e20270c98ef2d2fff0cf1fe30264751d4849d8d1be4'
    v10=torch.load(V10_PT,map_location='cpu',weights_only=False,mmap=True);v11=torch.load(V11_PT,map_location='cpu',weights_only=False,mmap=True)
    previous=torch.load(PREVIOUS_PT,map_location='cpu',weights_only=False,mmap=True)
    assert json.loads((V10_PT.parents[1]/'execution-exit.json').read_text())['returncode']==0
    assert json.loads((V11_PT.parents[1]/'scoring-exit.json').read_text())['returncode']==0
    previous_exit=json.loads((PREVIOUS_PT.parents[1]/'scoring-exit.json').read_text())
    assert previous_exit['returncode']==0 and previous_exit['predictions.pt_sha256']==sha(PREVIOUS_PT)
    assert previous_exit['final_checkpoint_sha256']=='fc193096180fa99969ba506811b2aeb041c927b895c2a49efd803e2c9442f431'
    refs=[]
    for label,sealed in (('v10-final',v10),('v11-final',v11),('v12-previous-r2',previous)):
        assert len(windows)==len(sealed['windows'])==2
        for w,original in zip(windows,sealed['windows']):
            assert w['plan']==original['plan'] and bits(w['mask'],original['mask']) and w['truth'].keys()==original['truth'].keys()
            assert all(bits(value,original['truth'][key]) for key,value in w['truth'].items())
        for item in sealed['predictions']:
            if label=='v11-final' and item['world_label']!='final':continue
            refs.append(dict(world_label=label,recording_index=item['recording_index'],mode=item['mode'],tensors=item['tensors']))
    assert len(refs)==12
    return refs,dict(v10=dict(path=str(V10_PT),sha256=sha(V10_PT)),v11=dict(path=str(V11_PT),sha256=sha(V11_PT)),
        previous_v12=dict(path=str(PREVIOUS_PT),sha256=sha(PREVIOUS_PT)))


from scripts.agent.supermimic._motion_scoring import predict, summarize as _summarize


def motion_arrays(pred,truth):
    p,t=pred['path'][...,-1,:],truth['path'][...,-1,:]
    qt,qp=unit_quat(t[...,3:7]),unit_quat(p[...,3:7]);inverse=torch.cat((qt[...,:1],-qt[...,1:]),-1)
    angle=rotation_vector(quat_mul(inverse,qp)).norm(dim=-1)*180/math.pi
    qidx=[i for i in range(7,25) if i not in (9,14,11,16)];vidx=[i for i in range(31,49) if i not in (35,40)]
    return dict(root_linear_velocity_m_s=(p[...,25:28],t[...,25:28]),root_angular_velocity_rad_s=(p[...,28:31],t[...,28:31]),
        LINK_vx_m_s=(pred['features'][...,0],truth['features'][...,0]),orientation_geodesic_degrees=(angle,torch.zeros_like(angle)),
        nonwheel_hinge_q_rad=(p[...,qidx],t[...,qidx]),slider_q_m=(p[...,[11,16]],t[...,[11,16]]),
        hinge_velocity_including_wheels_rad_s=(p[...,vidx],t[...,vidx]),slider_velocity_m_s=(p[...,[35,40]],t[...,[35,40]]),
        root_position_x_m=(p[...,0],t[...,0]),root_position_y_m=(p[...,1],t[...,1]),root_position_z_m=(p[...,2],t[...,2]),
        left_wheel_velocity_rad_s=(p[...,33],t[...,33]),right_wheel_velocity_rad_s=(p[...,38],t[...,38]),
        nonwheel_hinge_velocity_rad_s=(p[...,[i for i in vidx if i not in (33,38)]],t[...,[i for i in vidx if i not in (33,38)]]))


def summarize(pred,w):
    # 保留本版本的指标集合，不把不同实验的验收公式混用。
    return _summarize(pred,w,motion_arrays)


def decision(results,label):
    decisions=[]
    for index in (0,1):
        by_label={r['world_label']:r['groups']['all']['24'] for r in results if r['recording_index']==index and r['mode']=='free'}
        tests={};new=by_label[label]
        for metric in METRICS:
            value=new['metrics'][metric]['mae'];target=by_label['v12-previous-r2']['metrics'][metric]['mae']
            tests[metric]=dict(candidate=value,previous_v12_reference=target,no_worse=value is not None and target is not None and value<=target)
        strict=any(tests[name]['candidate'] is not None and tests[name]['previous_v12_reference'] is not None and tests[name]['candidate']<tests[name]['previous_v12_reference'] for name in ('LINK_vx_m_s','orientation_geodesic_degrees'))
        nf=new['retained_motion_nonfinite_values']+sum(v['nonfinite_prediction']+v['nonfinite_derived_computation'] for v in new['metrics'].values())
        passed=all(x['no_worse'] for x in tests.values()) and strict and nf==0
        decisions.append(dict(recording_index=index,metrics=tests,speed_or_pose_strictly_better=strict,nonfinite_count=nf,all_conditions_passed=passed))
    return dict(world_label=label,per_recording=decisions,joint_motion_forecast_screen_passed=all(d['all_conditions_passed'] for d in decisions),
        scope='Engineering H24 free motion forecast screen only; no policy/real reward/support or robot acceptance.')


def checkpoint_selection(label,path,run,rows):
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    row=rows.get(saved['iteration'])
    assert row is not None and row['world_updates']==saved['world_updates'] and row['transitions']==saved['transitions']
    proof=dict(label=label,path=str(path),sha256=sha(path),iteration=saved['iteration'],
        transitions=saved['transitions'],world_updates=saved['world_updates'],metrics_row=row)
    if label!='final':
        threshold={'w1708':1708,'w3008':3008}[label]
        candidates=[p for p in sorted(run.glob('checkpoint_*.pt'))
            if int(p.stem.removeprefix('checkpoint_')) in rows
            and rows[int(p.stem.removeprefix('checkpoint_'))]['world_updates']>=threshold]
        assert candidates and candidates[0].resolve()==path
        assert path.name==f"checkpoint_{saved['iteration']:06d}.pt"
        proof.update(selection='first published immutable numbered checkpoint reaching fixed world_updates',threshold=threshold)
    else:
        assert path.name=='final.pt'
        proof['selection']='actual final after successful real training exit'
    return proof


def load_world(label,path,current,infos,parent_actor):
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert saved['algorithm_contract']==V12 and saved['infos']==infos and saved['asset_sha256']==core.asset_digest()
    assert saved['config']['world_loss_recipe']==WORLD_LOSS_RECIPE and saved['config']['mode']=='train' and saved['config']['task']=='chuanliantui_walk'
    assert saved['initialization']==saved['config']['initialization']
    init=saved['initialization'];assert init['kind']=='actor_transfer_world_coldstart' and init['parent_sha256']==sha(PARENT)
    transfer=init['reward_only_actor_transfer']
    assert transfer['kind']=='SM615_reward_only_v5_to_v7_actor_transfer' and transfer['all_nonreward_infos_equal'] and transfer['world_critic_optimizers_not_inherited']
    assert set(transfer['removed_reward_terms'])==REMOVED_REWARD_TERMS and transfer['target_reward_weights']==infos['reward_weights']
    assert saved['actor'].keys()==parent_actor.keys() and all(bits(value,parent_actor[key]) for key,value in saved['actor'].items())
    assert json.loads((path.parent/'source_sha256.json').read_text())==current
    assert all(sha(path.parent/'snapshot/ct_mjlab'/name)==value for name,value in current.items())
    model=WorldModel().cpu().eval().requires_grad_(False);model.load_state_dict(saved['world'],strict=True)
    proof=dict(label=label,checkpoint=str(path),checkpoint_sha256=sha(path),world_state_sha256_before=core.state_digest(model),
        iteration=saved['iteration'],transitions=saved['transitions'],world_updates=saved['world_updates'],
        initialization=init,full_actor_equals_615=True,task_infos=infos,recipe=WORLD_LOSS_RECIPE)
    return model,proof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',nargs=2,action='append',metavar=('LABEL','PT'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exit-record',type=Path,default=ART/'train-exit.json');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);assert ALGORITHM_CONTRACT==V12 and sha(core.__file__)=='774de38163fe3524132f55ebd2636f8b345257ac7ba640a45636e157ff19f70d'
    entries=[(label,Path(path).resolve()) for label,path in args.checkpoint]
    assert len({label for label,path in entries})==len(entries) and len({path for label,path in entries})==len(entries)
    if not args.preflight:assert {label for label,path in entries}=={'w1708','w3008','final'}
    current,old,bridge=source_bridge();assert json.loads((ART/'training-source-sha256.json').read_text())==current
    records,infos,old_infos=recordings(old,current);windows,historical_reward=historical_windows(records)
    references,reference_files=load_references(windows)
    parent_actor=torch.load(PARENT,map_location='cpu',weights_only=False,mmap=True)['actor']
    run=entries[0][1].parent;assert run.name=='20261031_v12_motiononly_retrain_r1' and all(path.parent==run for label,path in entries)
    models=[];proofs=[]
    for label,path in entries:
        model,proof=load_world(label,path,current,infos,parent_actor);models.append((label,model));proofs.append(proof)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    snapshot=args.output/'snapshot';snapshot.mkdir();shutil.copy2(__file__,snapshot/Path(__file__).name)
    common=dict(source_role=bridge,current_world_sources=current,recordings=[r['proof'] for r in records],references=reference_files,
        historical_reward=historical_reward,current_infos=infos,historical_infos=old_infos)
    if args.preflight:
        assert all(core.state_digest(model)==proof['world_state_sha256_before'] for (_,model),proof in zip(models,proofs))
        core.write(args.output/'preflight.json',dict(all_checks_passed=True,worlds=proofs,proof=common,windows=[len(w['plan']) for w in windows],
            historical_truth_bitwise_equals_all_three_sealed_references=True,model_forwards=0,optimizer_steps=0,scorer_sha256=sha(__file__)))
        print(json.dumps(dict(preflight=True,all_checks_passed=True,model_forwards=0)));return
    receipt=json.loads(args.exit_record.read_text())
    final_path=dict(entries)['final'];assert receipt['returncode']==0 and receipt['final_checkpoint_exists']
    assert Path(receipt['log_dir']).resolve()==run and receipt['final_sha256']==sha(final_path)
    rows={}
    for line in (run/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line);rows[row['iteration']]=row
    selections=[checkpoint_selection(label,path,run,rows) for label,path in entries]
    plan_path=ART/'decision-plan.json';plan=json.loads(plan_path.read_text())
    assert plan['milestone_world_updates']==[1708,3008] and plan['metrics']==list(METRICS)
    assert 'previous V12 r2' in plan['screen'] and 'Final after real exit0 only' in plan['selection']
    predictions=[];checks=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No optimizer')))
        stack.enter_context(patch.object(RunningNorm,'update',side_effect=AssertionError('No norm update')))
        for label,model in models:
            for index,w in enumerate(windows):
                free=predict(model,w,False);teacher=predict(model,w,True)
                h1={name:bits(value[:,0],free[name][:,0]) for name,value in teacher.items()};assert all(h1.values())
                for mode,tensors in (('free',free),('teacher',teacher)):
                    assert torch.count_nonzero(tensors['path'][...,[9,14]+list(range(49,64))])==0
                    predictions.append(dict(world_label=label,recording_index=index,mode=mode,tensors=tensors))
                checks.append(dict(world_label=label,recording_index=index,h1_bitwise=h1))
    for (_,model),proof in zip(models,proofs):
        assert core.state_digest(model)==proof['world_state_sha256_before'];proof['world_state_sha256_after']=core.state_digest(model)
    sealed=args.output/'predictions.pt';temporary=args.output/'predictions.tmp'
    torch.save(dict(schema='motion-retrain-prediction-v1',windows=windows,predictions=predictions,worlds=proofs,checks=checks,
        reference_files=reference_files,proof=common,selections=selections),temporary);temporary.replace(sealed)
    seal_sha=sha(sealed);del predictions,windows,models
    artifact=torch.load(sealed,map_location='cpu',weights_only=False,mmap=True);results=[]
    for item in artifact['predictions']+references:
        index=item['recording_index'];results.append(dict(world_label=item['world_label'],recording_index=index,mode=item['mode'],**summarize(item['tensors'],artifact['windows'][index])))
    screens={label:decision(results,label) for label,path in entries};screen=screens['final']
    assert all(sha(Path(proof['checkpoint']))==proof['checkpoint_sha256'] for proof in proofs)
    assert source_bridge()[0]==current and sha(sealed)==seal_sha
    for record in records:
        for info in record['proof']['files'].values():assert sha(info['path'])==info['sha256']
    for info in reference_files.values():assert sha(info['path'])==info['sha256']
    report=dict(schema=artifact['schema'],worlds=proofs,proof=common,reference_files=reference_files,checks=checks,results=results,
        selections=selections,screens=screens,screen=screen,decision_plan_sha256=sha(plan_path),predictions_sha256=seal_sha,
        scorer_sha256=sha(__file__),training_exit_record=dict(path=str(args.exit_record.resolve()),sha256=sha(args.exit_record),record=receipt),
        optimizer_steps=0,normalization_updates=0,old_world_forward_calls=0,actor_forward_calls=0,simulator_steps=0,
        model_reward_not_real_task=True,old_record_reward_identity='Historical v5 18-term labels reconstructed by archived real functions; never relabeled as current v7 13-term.',
        excluded_prediction_channels='Wheel q and auxiliary torque/contact slots in new path64 are constant zeros, never scored as predictions.')
    core.write(args.output/'score.json',report)
    lines=['# 运动预测模型重新训练同轨迹比较','',
        '旧18项轨迹保持原身份，仅比较共同运动量；当前模型属于13项任务配方，评分阶段零参数更新，未重新运行旧模型。','',
        '| 模型/轨迹/模式 | H24 LINK MAE m/s | 姿态MAE ° | 非轮转动关节q MAE rad | 滑动q MAE m |','|---|---:|---:|---:|---:|']
    def number(v):return '—' if v is None else f'{v:.6f}'
    for row in results:
        m=row['groups']['all']['24']['metrics'];lines.append(f"| {row['world_label']}/{row['recording_index']}/{row['mode']} | "+' | '.join(number(m[k]['mae']) for k in ('LINK_vx_m_s','orientation_geodesic_degrees','nonwheel_hinge_q_rad','slider_q_m'))+' |')
    lines+=['','最终模型两轨迹H24八项不劣上一轮V12 r2、且速度或姿态至少一项严格改善：'+str(screen['joint_motion_forecast_screen_passed'])+'。',
        '两个预定更新数模型独立报告，未以早期最好点替代final；V10/V11仅作独立参考。','',
        'H1/8/24、13组、24拍曲线及root XYZ、左右轮速和非轮转动关节速度见JSON。线速度world系、root角速度body系；关节位置/速度分别保留rad/m与rad/s/m/s。',
        'teacher H24是该目标时刻的单拍预测，free展开240ms，LINK目标238ms。','',
        '训练数据、更新数和任务配方与历史模型不同，不作严格单因素因果结论；不证明策略、原奖励等价、承重、reset或机器人行为通过。']
    (args.output/'score.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output),predictions_sha256=seal_sha,joint_motion_forecast_screen_passed=screen['joint_motion_forecast_screen_passed'])))


if __name__=='__main__':main()
