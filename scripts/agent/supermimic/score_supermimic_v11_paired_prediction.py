#!/usr/bin/env python3
"""Frozen V11 worlds on the original V10 recordings and V9 controllers.

Only source-role/identity loading is adapted. Prediction, masking, physical
metrics, rewards and teacher-state replacement reuse the frozen original tools.
"""
import argparse
import ast
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from scripts.agent.supermimic import score_supermimic_physical_prediction as core
from scripts.agent.supermimic import diagnose_supermimic_rollout_drift as drift
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, WORLD_LOSS_RECIPE

ROOT = core.ROOT
V11 = 'chuanliantui-supermimic-canonical-physics-fixed-support-scale-robust-wheel-angle-v11'
OLD_TRIAL = ROOT/'artifacts/supermimic/supermimic-v10-world-only-20261031-r1'
REFERENCE = ROOT/'artifacts/supermimic/supermimic-v10-one-step-vs-rollout-20261031-r3'
NEW_TRIAL = ROOT/'artifacts/supermimic/supermimic-v11-world-only-20261031-r1'
OLD_RUN = ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v10_contact_worldonly_r1'
LOCKS = dict(
    core='774de38163fe3524132f55ebd2636f8b345257ac7ba640a45636e157ff19f70d',
    teacher='40caf7750874c72de9244b3774ab0578a23a3798bb9617255d5852632523e685',
    old_manifest='f9297b547d969d5226e81de5d8cd678faf28f2ea8bc4fc99790f356852609c86',
    reference_predictions='b0646cd49a6f019b7ddf2cf0735a8b085230be1ff064c858fb25fffac08d3eb2',
    reference_score='3caf13e27ed9bd0836c0521522b702a03f758a90bd6f1afa1d8a051c682954b7',
    reference_summary='cc59c2a6f7f5276649b815f656bec6eac605b3ac624d2dade734b613694f4596')
SOURCE_PAIRS = {
    'supermimic/__main__.py': ('16533a9419e4db0d39c0b45fbdc3b28360439a9ac9d6082faf0618ba38880e9b',
        'bbcc213b28f46be8a95d8fc80dda394f3077ad70d9b45f34539a864f7571fb4a'),
    'supermimic/learner.py': ('cc007167bea0d4ce36e9bef343633592364dea13298903c63245da44847e6867',
        'a51da28f583711caf2ef6f71216ded99d2fa0d4051ed06893f6d7058937433c6')}
EXPECTED_RECIPE = dict(version='fixed-support-scale-robust-wheel-angle-v1',
    wheel_vertical_force_scale_N=5., wheel_position_penalty='twice_huber',
    wheel_position_huber_delta=1., other_coordinates='squared_delta_std',
    output_delta_normalization='unchanged_v10')


def definition(path, names):
    nodes = ast.parse(path.read_text()).body
    for name in names:
        node = next(x for x in nodes if getattr(x, 'name', None) == name)
        nodes = getattr(node, 'body', [])
    return ast.dump(node, include_attributes=False)


def source_bridge(current):
    if ALGORITHM_CONTRACT != V11 or WORLD_LOSS_RECIPE != EXPECTED_RECIPE:
        raise ValueError('Current V11 contract/recipe differs from the reviewed candidate')
    if core.digest(Path(core.__file__)) != LOCKS['core'] or core.digest(Path(drift.__file__)) != LOCKS['teacher']:
        raise ValueError('Original scoring or teacher implementation changed')
    manifest = OLD_RUN/'source_sha256.json'
    if core.digest(manifest) != LOCKS['old_manifest']:
        raise ValueError('Original V10 source manifest changed')
    old = json.loads(manifest.read_text())
    if len(old) != 99 or old.keys() != current.keys():
        raise ValueError('Require the same99 source filenames')
    changes = {}
    for name, sha in old.items():
        if core.digest(OLD_RUN/'snapshot/ct_mjlab'/name) != sha:
            raise ValueError(f'Original V10 snapshot changed: {name}')
        if current[name] != sha:
            if SOURCE_PAIRS.get(name) != (sha, current[name]):
                raise ValueError(f'Unexpected capture/world source difference: {name}')
            changes[name] = dict(capture_v10_sha256=sha, world_v11_sha256=current[name])
    if set(changes) != set(SOURCE_PAIRS):
        raise ValueError('Require exactly the reviewed main/learner source changes')
    functions = [('__main__.py', ('collect',)), ('__main__.py', ('make_env',)),
        ('__main__.py', ('supermimic_nominal_contact',)), ('__main__.py', ('validate_contact_encoding',)),
        ('networks.py', ('make_actor',)), ('networks.py', ('act',)),
        ('networks.py', ('WorldModel', 'inputs')), ('networks.py', ('WorldModel', 'step')),
        ('state.py', ('physical_snapshot',)), ('state.py', ('capture_state',)),
        ('state.py', ('capture_parameters',))]
    ast_proof = {}
    for module, names in functions:
        a = OLD_RUN/'snapshot/ct_mjlab/supermimic'/module
        b = ROOT/'mjlab_training/src/ct_mjlab/supermimic'/module
        key = module+':'+'.'.join(names)
        ast_proof[key] = definition(a, names) == definition(b, names)
    if not all(ast_proof.values()):
        raise ValueError('Capture, action or world inference executable path changed')
    return old, dict(capture_manifest=str(manifest), capture_manifest_sha256=core.digest(manifest),
        capture_contract=core.V10, world_contract=V11, controller_contract=core.V9,
        unchanged_source_files=97, differences=changes, executable_ast_equal=ast_proof,
        learner_load_note='Full Learner.load changed; no new controller load or capture is performed. Original V9 controller identity is verified by the frozen V10 loader.')


def load_capture(directory, old, current, infos):
    """Keep the original ROOT hash check, using its exact historical source tree."""
    expected = OLD_TRIAL/directory.name
    if directory.resolve() != expected or directory.name not in ('midpoint-actor', 'final-actor'):
        raise ValueError('Only the two original V10 recording directories are allowed')
    sources = json.loads((directory/'source-sha256.json').read_text())
    prefix = 'mjlab_training/src/ct_mjlab/'
    if {k[len(prefix):]: v for k, v in sources.items() if k.startswith(prefix)} != old:
        raise ValueError('Recording source99 does not match the locked V10 manifest')
    for name, sha in sources.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe archived source path')
        if core.digest(directory/'snapshot'/relative) != sha:
            raise ValueError(f'Capture archive changed: {name}')
        if core.digest(ROOT/relative) != sha:
            key = name[len(prefix):] if name.startswith(prefix) else None
            if key not in SOURCE_PAIRS or SOURCE_PAIRS[key] != (sha, current[key]):
                raise ValueError(f'Unreviewed current/capture difference: {name}')
    # No digest is faked and no guard is removed. The original loader checks the
    # actual103 recorded files against their actual archived ROOT, then restores it.
    with patch.object(core, 'ROOT', directory/'snapshot'):
        record = core.load_recording(directory, old, infos)
    record['proof']['source_role_verification'] = dict(verified=True,
        capture_source_root=str(directory/'snapshot'), current_world_source_root=str(ROOT),
        permitted_current_differences=sorted(SOURCE_PAIRS), recorded_labels_unchanged=True)
    return record


def check_world_header(saved, infos):
    if (saved.get('algorithm_contract') != V11 or saved.get('asset_sha256') != core.asset_digest()
            or saved.get('infos') != infos or saved.get('interrupted', False)
            or saved.get('config', {}).get('mode') != 'train'
            or saved.get('config', {}).get('task') != 'chuanliantui_walk'
            or saved.get('config', {}).get('world_steps_per_update') != 4
            or saved.get('config', {}).get('world_loss_recipe') != EXPECTED_RECIPE):
        raise ValueError('World must have the exact V11 identity/XML/infos/recipe')


def load_world(label, checkpoint, current, infos):
    if not checkpoint.is_absolute() or checkpoint.suffix != '.pt' or checkpoint.name.endswith('.tmp'):
        raise ValueError('Require a published absolute.pt checkpoint')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    check_world_header(saved, infos)
    source = core.verified_training_sources(checkpoint, current)
    world = core.WorldModel().cpu().eval().requires_grad_(False)
    world.load_state_dict(saved['world'], strict=True)
    if any(v.is_floating_point() and not v.isfinite().all() for v in world.state_dict().values()):
        raise ValueError('Nonfinite world tensor')
    return dict(label=label, world=world, proof=dict(label=label, checkpoint=str(checkpoint),
        checkpoint_sha256=core.digest(checkpoint), algorithm_contract=V11,
        asset_sha256=saved['asset_sha256'], world_loss_recipe=saved['config']['world_loss_recipe'],
        iteration=saved['iteration'], world_updates=saved['world_updates'], training_sources=source,
        world_state_sha256_before=core.state_digest(world)))


def load_reference(directory):
    if directory.resolve() != REFERENCE:
        raise ValueError('Require the original r3 reference directory')
    receipt = json.loads((directory/'execution-exit.json').read_text())
    terminal = json.loads((directory/'terminal-exit.json').read_text())
    targets = {'reference_predictions': directory/'result/predictions.pt',
        'reference_score': directory/'result/diagnostic.json',
        'reference_summary': directory/'result/core_summary.json'}
    if receipt.get('returncode') != 0 or terminal.get('exit_code') != 0:
        raise ValueError('Reference lacks actual exit0 receipts')
    for name, path in targets.items():
        if core.digest(path) != LOCKS[name] or receipt.get(path.name+'_sha256') != LOCKS[name]:
            raise ValueError(f'Reference identity changed: {name}')
    sealed = torch.load(targets['reference_predictions'], map_location='cpu', weights_only=False, mmap=True)
    if sealed.get('schema') != 'v10-true-state-versus-free-rollout-v1':
        raise ValueError('Require the singular-world r3 teacher/free schema')
    if sealed['world']['checkpoint_sha256'] != 'aed433cc4bd49f43f71c24a735d32375cc0be41d2cd82adbf21214334a03b304':
        raise ValueError('Reference is not the original V10 final')
    if core.digest(Path(sealed['world']['checkpoint'])) != sealed['world']['checkpoint_sha256']:
        raise ValueError('Original V10 reference checkpoint changed')
    return sealed, json.loads(targets['reference_summary'].read_text()), targets


def same_windows(actual, expected):
    return (actual['plan'] == expected['plan'] and drift.tensor_equal(actual['mask'], expected['mask'])
        and actual['truth'].keys() == expected['truth'].keys()
        and all(drift.tensor_equal(v, expected['truth'][k]) for k, v in actual['truth'].items()))


def decision(result, reference):
    a, b = result['groups']['all']['24'], reference['groups']['all']['24']
    rules = [('LINK_speed_lower', 'reward_link_vx_m_s', True),
        ('orientation_lower', 'rotation_geodesic_deg', True),
        ('left_support_no_higher', 'support_left_N', False),
        ('right_support_lower', 'support_right_N', True),
        ('total_support_bodyweight_no_higher', 'support_total_bodyweight', False)]
    checks = {}
    for name, metric, strict in rules:
        new, old = a['metrics'][metric]['mae'], b['metrics'][metric]['mae']
        checks[name] = dict(candidate=new, reference=old,
            passed=new is not None and old is not None and (new < old if strict else new <= old))
    new, old = a['hard_failure_confusion']['fp'], b['hard_failure_confusion']['fp']
    checks['hard_failure_fp_no_higher'] = dict(candidate=new, reference=old, passed=new <= old)
    counts = {k:v['nonfinite_available_scalar_values'] for k,v in a['raw_nonfinite'].items() if k.startswith('prediction/')}
    counts.update({'metric/'+k:v['nonfinite_prediction'] for k,v in a['metrics'].items()})
    counts.update({'cumulative_reward/'+k:v['nonfinite_prediction']
        for k,v in a['cumulative_full_window_reward'].items()})
    checks['all_prediction_nonfinite_zero'] = dict(counts=counts, passed=all(v == 0 for v in counts.values()))
    return dict(checks=checks, all_conditions_passed=all(x['passed'] for x in checks.values()),
        scope='Fixed nominal H24 free prediction criteria, not training loss, actor learning or behavior acceptance.')


def markdown(report):
    def number(value, digits=6):
        return '—' if value is None else f'{value:.{digits}f}'
    lines = ['# V11同数据世界模型预测对照', '',
        '控制器V9、录制V10、世界模型V11身份分开；真实数据/动作/窗口未改，本评分阶段不新采集或更新参数。', '',
        '| 模型 | 轨迹 | 模式 | H24 LINK速度MAE m/s | 姿态MAE ° | 左/右支撑MAE N |',
        '|---|---|---|---:|---:|---:|']
    for r in report['results']:
        m=r['groups']['all']['24']['metrics']
        lines.append(f"| {r['world_label']} | {Path(r['recording']).name} | {r['mode']} | "+
            number(m['reward_link_vx_m_s']['mae'])+' | '+number(m['rotation_geodesic_deg']['mae'],4)+' | '+
            number(m['support_left_N']['mae'],4)+'/'+number(m['support_right_N']['mae'],4)+' |')
    lines += ['', 'H1/8/24、24拍曲线、all/parking/case/阶段/context分组、奖励与失败混淆见JSON。teacher在每拍输入真实state，H24仍是该时刻的一拍条件预测；free才是连续240ms滚动。', '',
        '预声明条件逐数据对照V10 final，early/final分别报告；新训练loss下降不代表预测改善。误差差值不是可加因果分解。']
    return '\n'.join(lines)+'\n'


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    if args.trial.resolve() != OLD_TRIAL or args.batch != 64:
        raise ValueError('Require the original V10 trial and batch64')
    current = core.current_training_sources()
    old, bridge = source_bridge(current)
    infos = core.contracts(core.command_cfg('walk'))
    reference, reference_summary, reference_files = load_reference(args.reference)
    records = [load_capture(OLD_TRIAL/name, old, current, infos) for name in ('midpoint-actor','final-actor')]
    for i,record in enumerate(records):
        if record['proof']['files'] != reference['recordings'][i]['files']:
            raise ValueError('Recording file identity differs from the original sealed reference')
    windows = [core.make_windows(r) for r in records]
    for i,w in enumerate(windows):
        if not same_windows(w, reference['windows'][i]):
            raise ValueError('Original fixed windows/truth/mask changed')
    if len(args.checkpoint) != 2 or {p[0] for p in args.checkpoint} != {'early', 'final'}:
        raise ValueError('Require exactly one early and one final world')
    models = [load_world(label, Path(path), current, infos) for label,path in args.checkpoint]
    if len({m['proof']['checkpoint_sha256'] for m in models}) != 2:
        raise ValueError('Require two distinct V11 checkpoint identities')
    if len({r['proof']['actor_checkpoint_sha256'] for r in records}) != 2:
        raise ValueError('Require the two distinct recorded V9 controllers')
    plan_path = NEW_TRIAL/'decision-plan.json'
    plan = json.loads(plan_path.read_text())
    expected_conditions = ['H24 free LINK speed MAE lower', 'H24 free orientation MAE lower',
        'H24 free left support MAE no higher', 'H24 free right support MAE lower',
        'H24 total support/bodyweight MAE no higher', 'H24 hard-failure FP no higher', 'nonfinite predictions zero']
    if plan.get('per_recording_conditions') != expected_conditions:
        raise ValueError('Decision plan does not match the declared same-data criteria')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    scripts = {str(Path(p).resolve().relative_to(ROOT)):core.digest(Path(p)) for p in (__file__,core.__file__,drift.__file__)}
    for name,sha in scripts.items():
        dest=args.output/'snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
        if core.digest(dest) != sha: raise ValueError('Scoring source snapshot changed')
    core.write(args.output/'source-role-proof.json',dict(bridge=bridge, current_world_sources=current,
        capture_v10_sources=old, scripts=scripts, decision_plan=dict(path=str(plan_path),sha256=core.digest(plan_path),contents=plan),
        reference_files={k:dict(path=str(p),sha256=core.digest(p)) for k,p in reference_files.items()}))
    predictions, checks, oracles = [], [], []
    with ExitStack() as guard:
        guard.enter_context(patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No optimizer')))
        guard.enter_context(patch.object(core.RunningNorm,'update',side_effect=AssertionError('No norm update')))
        for record in records:
            oracles.append(core.oracle_check(models[0]['world'].adapter,record))
        for model in models:
            for i,w in enumerate(windows):
                free = core.predict(model['world'], w, args.batch)
                teacher,calls = drift.teacher_predict(model['world'],w,args.batch)
                h1={k:drift.tensor_equal(v[:,0],free[k][:,0]) for k,v in teacher.items()}
                if not all(h1.values()): raise AssertionError('Teacher/free H1 mismatch')
                checks.append(dict(world_label=model['label'],recording_index=i,h1_outputs_bitwise_equal=h1,
                    teacher_world_step_calls=calls,free_world_step_calls=calls))
                for mode,tensors in (('free',free),('teacher',teacher)):
                    predictions.append(dict(world_label=model['label'],recording_index=i,mode=mode,tensors=tensors))
    for model in models:
        model['proof']['world_state_sha256_after']=core.state_digest(model['world'])
        if model['proof']['world_state_sha256_after'] != model['proof']['world_state_sha256_before']:
            raise AssertionError('World/norm changed')
    artifact=dict(schema='v11-same-v10-recordings-teacher-free-v1',windows=windows,predictions=predictions,
        worlds=[m['proof'] for m in models],recordings=[r['proof'] for r in records],checks=checks,oracle=oracles,
        reference=dict(world=reference['world'],source_predictions_sha256=LOCKS['reference_predictions'],
            summaries=reference_summary['results']),source_role_proof_sha256=core.digest(args.output/'source-role-proof.json'))
    temporary=args.output/'predictions.tmp';sealed=args.output/'predictions.pt'
    torch.save(artifact,temporary);temporary.replace(sealed);sealed_sha=core.digest(sealed)
    del artifact,predictions,windows,reference
    artifact=torch.load(sealed,map_location='cpu',weights_only=False)
    results,decisions=[],[]
    for prediction in artifact['predictions']:
        i=prediction['recording_index'];summary=core.summarize(prediction['tensors'],artifact['windows'][i])
        row=dict(world_label=prediction['world_label'],recording=str(records[i]['directory']),mode=prediction['mode'],**summary)
        results.append(row)
        if prediction['mode']=='free':
            baseline=next(x for x in artifact['reference']['summaries'] if Path(x['recording']).name==records[i]['directory'].name and x['mode']=='free')
            decisions.append(dict(world_label=prediction['world_label'],recording=str(records[i]['directory']),**decision(summary,baseline)))
    for model in models:
        if core.digest(Path(model['proof']['checkpoint'])) != model['proof']['checkpoint_sha256']:
            raise AssertionError('World checkpoint changed')
        core.verified_training_sources(Path(model['proof']['checkpoint']),current)
    for record in records:
        for info in record['proof']['files'].values():
            if core.digest(Path(info['path'])) != info['sha256']: raise AssertionError('Recording changed')
        for name,sha in record['proof']['sources'].items():
            if core.digest(record['directory']/'snapshot'/name)!=sha: raise AssertionError('Capture archive changed')
            relative_name=name.removeprefix('mjlab_training/src/ct_mjlab/')
            expected_current=SOURCE_PAIRS[relative_name][1] if relative_name in SOURCE_PAIRS else sha
            if core.digest(ROOT/name)!=expected_current: raise AssertionError('Current capture dependency changed')
        if core.digest(Path(record['proof']['metadata']['checkpoint'])) != record['proof']['actor_checkpoint_sha256']:
            raise AssertionError('V9 controller checkpoint changed')
    for name,sha in scripts.items():
        if core.digest(ROOT/name)!=sha: raise AssertionError('Scoring implementation changed')
    source_bridge(core.current_training_sources())
    if core.current_training_sources()!=current or core.digest(sealed)!=sealed_sha:
        raise AssertionError('Current source or sealed predictions changed')
    for name,path in reference_files.items():
        if core.digest(path)!=LOCKS[name]: raise AssertionError('Reference changed')
    report=dict(schema=artifact['schema'],score_script_sha256=scripts[str(Path(__file__).resolve().relative_to(ROOT))],
        worlds=artifact['worlds'],recordings=artifact['recordings'],reference=artifact['reference'],results=results,
        decisions=decisions,per_model_joint_improvement={label:all(x['all_conditions_passed'] for x in decisions if x['world_label']==label) for label in ('early','final')},
        oracle=oracles,checks=checks,predictions_sha256=sealed_sha,
        source_role_proof_sha256=artifact['source_role_proof_sha256'],cpu_only=True,optimizer_steps=0,
        normalization_updates=0,actor_forward_calls=0,critic_forward_calls=0,simulator_steps=0)
    core.write(args.output/'score.json',report)
    (args.output/'score.md').write_text(markdown(report))
    print(json.dumps(dict(output=str(args.output),predictions_sha256=sealed_sha,pairs=4,modes=2,
        per_model_joint_improvement=report['per_model_joint_improvement'],optimizer_steps=0,simulator_steps=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--checkpoint',nargs=2,action='append',required=True,metavar=('LABEL','PT'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=64)
    run(parser.parse_args())


if __name__=='__main__':
    main()
