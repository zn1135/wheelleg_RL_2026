#!/usr/bin/env python3
"""V12 motion-only forecasts versus sealed V10/V11 common motion quantities."""
import argparse
import ast
from contextlib import ExitStack
import hashlib
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
ART=ROOT/'artifacts/supermimic/supermimic-v12-motion-only-20261031-r2'
V12='chuanliantui-supermimic-motion-only-pd-reward-soft-termination-v12'
OLD=ROOT/'artifacts/supermimic/supermimic-v10-world-only-20261031-r1'
OLD_RUN=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v10_contact_worldonly_r1'
V10_PT=ROOT/'artifacts/supermimic/supermimic-v10-one-step-vs-rollout-20261031-r3/result/predictions.pt'
V11_PT=ROOT/'artifacts/supermimic/supermimic-v11-world-only-20261031-r1/score/predictions.pt'
LOCKED_NEW={
    'supermimic/__main__.py':'bea771fd2e0686b53a196c78f5d488df95471ae1878a56b6f6d01502ddcb809c',
    'supermimic/learner.py':'fb27172a723ddf20c7e9f0a40993b10c97c38e1b9771e0b101708e7fb006b330',
    'supermimic/motion.py':'1ce8e3a6332f284b21d3db5f181fd502baa3359bc8048aad0b25a0c9f5e449b9',
    'supermimic/motion_adapter.py':'ceeef2f00c831c6a858bdad2080528de5436a789f51e73fabb4d18feaf557097'}
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
    return current,old,dict(world_source_count=101,capture_source_count=99,unchanged_sources=97,
        reviewed_changes_and_additions=LOCKED_NEW,collect_ast_equal=True,
        make_env_difference='Additional validate_motion_pd_config only; true task/physics sources remain unchanged.')


def recordings(old,current):
    infos=core.contracts(core.command_cfg('walk'));records=[]
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
        with patch.object(core,'ROOT',directory/'snapshot'):record=core.load_recording(directory,old,infos)
        records.append(record)
    return records,infos


def load_references(windows):
    assert sha(V10_PT)=='b0646cd49a6f019b7ddf2cf0735a8b085230be1ff064c858fb25fffac08d3eb2'
    assert sha(V11_PT)=='96a3474db724acf79603b794d447f1daf8cf4e51fd701403a9a1a87abdff021b'
    v10=torch.load(V10_PT,map_location='cpu',weights_only=False,mmap=True);v11=torch.load(V11_PT,map_location='cpu',weights_only=False,mmap=True)
    assert json.loads((V10_PT.parents[1]/'execution-exit.json').read_text())['returncode']==0
    assert json.loads((V11_PT.parents[1]/'scoring-exit.json').read_text())['returncode']==0
    refs=[]
    for label,sealed in (('v10-final',v10),('v11-final',v11)):
        for w,original in zip(windows,sealed['windows']):
            assert w['plan']==original['plan'] and bits(w['mask'],original['mask']) and w['truth'].keys()==original['truth'].keys()
            assert all(bits(value,original['truth'][key]) for key,value in w['truth'].items())
        for item in sealed['predictions']:
            if label=='v11-final' and item['world_label']!='final':continue
            refs.append(dict(world_label=label,recording_index=item['recording_index'],mode=item['mode'],tensors=item['tensors']))
    assert len(refs)==8
    return refs,dict(v10=dict(path=str(V10_PT),sha256=sha(V10_PT)),v11=dict(path=str(V11_PT),sha256=sha(V11_PT)))


from scripts.agent.supermimic._motion_scoring import predict, summarize as _summarize


def motion_arrays(pred,truth):
    p,t=pred['path'][...,-1,:],truth['path'][...,-1,:]
    qt,qp=unit_quat(t[...,3:7]),unit_quat(p[...,3:7]);inverse=torch.cat((qt[...,:1],-qt[...,1:]),-1)
    angle=rotation_vector(quat_mul(inverse,qp)).norm(dim=-1)*180/math.pi
    qidx=[i for i in range(7,25) if i not in (9,14,11,16)];vidx=[i for i in range(31,49) if i not in (35,40)]
    return dict(root_linear_velocity_m_s=(p[...,25:28],t[...,25:28]),root_angular_velocity_rad_s=(p[...,28:31],t[...,28:31]),
        LINK_vx_m_s=(pred['features'][...,0],truth['features'][...,0]),orientation_geodesic_degrees=(angle,torch.zeros_like(angle)),
        nonwheel_hinge_q_rad=(p[...,qidx],t[...,qidx]),slider_q_m=(p[...,[11,16]],t[...,[11,16]]),
        hinge_velocity_including_wheels_rad_s=(p[...,vidx],t[...,vidx]),slider_velocity_m_s=(p[...,[35,40]],t[...,[35,40]]))


def summarize(pred,w):
    # 保留本版本的指标集合，不把不同实验的验收公式混用。
    return _summarize(pred,w,motion_arrays)


def decision(results):
    decisions=[]
    for index in (0,1):
        by_label={r['world_label']:r['groups']['all']['24'] for r in results if r['recording_index']==index and r['mode']=='free'}
        tests={};new=by_label['v12-final']
        for metric in METRICS:
            value=new['metrics'][metric]['mae'];old=[by_label[label]['metrics'][metric]['mae'] for label in ('v10-final','v11-final')]
            target=min(old) if all(x is not None for x in old) else None
            tests[metric]=dict(candidate=value,best_reference=target,no_worse=value is not None and target is not None and value<=target)
        strict=any(tests[name]['candidate'] is not None and tests[name]['best_reference'] is not None and tests[name]['candidate']<tests[name]['best_reference'] for name in ('LINK_vx_m_s','orientation_geodesic_degrees'))
        nf=new['retained_motion_nonfinite_values']+sum(v['nonfinite_prediction']+v['nonfinite_derived_computation'] for v in new['metrics'].values())
        passed=all(x['no_worse'] for x in tests.values()) and strict and nf==0
        decisions.append(dict(recording_index=index,metrics=tests,speed_or_pose_strictly_better=strict,nonfinite_count=nf,all_conditions_passed=passed))
    return dict(per_recording=decisions,joint_motion_forecast_screen_passed=all(d['all_conditions_passed'] for d in decisions),
        scope='Engineering H24 free motion forecast screen only; no policy/real reward/support or robot acceptance.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exit-record',type=Path,default=ART/'train-exit.json');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);assert ALGORITHM_CONTRACT==V12 and sha(core.__file__)=='774de38163fe3524132f55ebd2636f8b345257ac7ba640a45636e157ff19f70d'
    current,old,bridge=source_bridge();records,infos=recordings(old,current);windows=[core.make_windows(r) for r in records]
    references,reference_files=load_references(windows);checkpoint=args.checkpoint.resolve();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['algorithm_contract']==V12 and saved['infos']==infos and saved['asset_sha256']==core.asset_digest()
    assert saved['config']['world_loss_recipe']==WORLD_LOSS_RECIPE and saved['config']['mode']=='train' and saved['config']['task']=='chuanliantui_walk'
    assert json.loads((checkpoint.parent/'source_sha256.json').read_text())==current
    assert all(sha(checkpoint.parent/'snapshot/ct_mjlab'/name)==value for name,value in current.items())
    if not args.preflight:
        receipt=json.loads(args.exit_record.read_text());assert receipt['returncode']==0 and receipt['final_checkpoint_exists'] and checkpoint.name=='final.pt'
        assert Path(receipt['log_dir'])==checkpoint.parent and receipt['final_sha256']==sha(checkpoint)
    model=WorldModel().cpu().eval().requires_grad_(False);model.load_state_dict(saved['world'],strict=True);before=core.state_digest(model)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    proof=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),world_state_sha256_before=before,source_role=bridge,
        current_world_sources=current,recordings=[r['proof'] for r in records],references=reference_files,recipe=WORLD_LOSS_RECIPE)
    if args.preflight:
        core.write(args.output/'preflight.json',dict(all_checks_passed=True,proof=proof,windows=[len(w['plan']) for w in windows],model_forwards=0))
        print(json.dumps(dict(preflight=True,all_checks_passed=True,model_forwards=0)));return
    plan_path=ART/'decision-plan.json';plan=json.loads(plan_path.read_text());assert plan['selection'].startswith('Actual final.pt')
    assert plan['metrics']==list(METRICS) and set(plan['excluded_metrics'])=={'predicted_torque','contact_force','wheel_support','accumulated_wheel_angle'}
    predictions=[];checks=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No optimizer')))
        stack.enter_context(patch.object(RunningNorm,'update',side_effect=AssertionError('No norm update')))
        for index,w in enumerate(windows):
            free=predict(model,w,False);teacher=predict(model,w,True)
            h1={name:bits(value[:,0],free[name][:,0]) for name,value in teacher.items()};assert all(h1.values())
            for mode,tensors in (('free',free),('teacher',teacher)):
                assert torch.count_nonzero(tensors['path'][...,[9,14]+list(range(49,64))])==0
                predictions.append(dict(world_label='v12-final',recording_index=index,mode=mode,tensors=tensors))
            checks.append(dict(recording_index=index,h1_bitwise=h1))
    assert core.state_digest(model)==before;proof['world_state_sha256_after']=before
    sealed=args.output/'predictions.pt';temporary=args.output/'predictions.tmp'
    torch.save(dict(schema='motion-only-prediction-v1',windows=windows,predictions=predictions,world=proof,checks=checks,reference_files=reference_files),temporary);temporary.replace(sealed)
    seal_sha=sha(sealed);del predictions,windows
    artifact=torch.load(sealed,map_location='cpu',weights_only=False);results=[]
    for item in artifact['predictions']+references:
        index=item['recording_index'];results.append(dict(world_label=item['world_label'],recording_index=index,mode=item['mode'],**summarize(item['tensors'],artifact['windows'][index])))
    screen=decision(results)
    assert sha(checkpoint)==proof['checkpoint_sha256'] and source_bridge()[0]==current and sha(sealed)==seal_sha
    for record in records:
        for info in record['proof']['files'].values():assert sha(info['path'])==info['sha256']
    for info in reference_files.values():assert sha(info['path'])==info['sha256']
    snapshot=args.output/'snapshot';snapshot.mkdir();shutil.copy2(__file__,snapshot/Path(__file__).name)
    report=dict(schema=artifact['schema'],world=proof,reference_files=reference_files,checks=checks,results=results,screen=screen,
        decision_plan_sha256=sha(plan_path),predictions_sha256=seal_sha,scorer_sha256=sha(__file__),optimizer_steps=0,normalization_updates=0,
        old_world_forward_calls=0,actor_forward_calls=0,simulator_steps=0,model_reward_not_real_task=True,
        excluded_prediction_channels='Wheel q and auxiliary torque/contact slots in new path64 are constant zeros, never scored as predictions.')
    core.write(args.output/'score.json',report)
    lines=['# 运动模型同轨迹比较','', '轮角位置、力矩和接触力已从新网络中删除；下表只比较共同运动量，未重新运行旧模型。','',
        '| 模型/轨迹/模式 | H24 LINK MAE m/s | 姿态MAE ° | 非轮转动关节q MAE rad | 滑动q MAE m |','|---|---:|---:|---:|---:|']
    def number(v):return '—' if v is None else f'{v:.6f}'
    for row in results:
        m=row['groups']['all']['24']['metrics'];lines.append(f"| {row['world_label']}/{row['recording_index']}/{row['mode']} | "+' | '.join(number(m[k]['mae']) for k in ('LINK_vx_m_s','orientation_geodesic_degrees','nonwheel_hinge_q_rad','slider_q_m'))+' |')
    lines+=['','两轨迹H24八项不劣各自最佳旧参考、且速度或姿态至少一项严格改善：'+str(screen['joint_motion_forecast_screen_passed'])+'。',
        '','H1/8/24、13组、24拍曲线、root线/角速度、16转动速度(含轮)和2滑动速度及完整分母/非有限数见JSON。teacher H24仍是该目标时刻的单拍预测；free为240ms展开，LINK为238ms。',
        '','新模型PD奖励和软终止有意不同于真实任务；不评价力/承重，不证明策略、原奖励、行为或真机通过。']
    (args.output/'score.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(output=str(args.output),predictions_sha256=seal_sha,joint_motion_forecast_screen_passed=screen['joint_motion_forecast_screen_passed'])))


if __name__=='__main__':main()
