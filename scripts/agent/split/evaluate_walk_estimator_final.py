#!/usr/bin/env python3
"""One fixed terminal encoder on two new complete runs; no training or control."""

import argparse
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from rsl_rl.modules import MLP

from scripts.agent.split import train_walk_estimator_only as training_source
from scripts.agent.split import walk_estimator_offline as offline


CANDIDATE_SHA = 'eeac74b114132b1e2d6781bca7a0864e287897abb4a7dedcb2e2d5aa34da7216'


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def bitwise_equal(left,right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and left.shape == right.shape and left.detach().cpu().contiguous().numpy().tobytes() == right.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(left,np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
    if isinstance(left,dict):
        return left.keys() == right.keys() and all(bitwise_equal(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):
        return len(left) == len(right) and all(bitwise_equal(a,b) for a,b in zip(left,right))
    if isinstance(left,float):
        return np.float64(left).tobytes() == np.float64(right).tobytes()
    return left == right


def check_protocol(meta,spec,card):
    assert meta['seed'] == spec['seed'] and meta['seconds'] == spec['seconds'] == 30
    assert meta['height'] == card['future_final_common']['height_m']
    assert meta['dt'] == .002 and meta['policy_dt'] == .01
    assert meta['feedback_mode'] == meta['velocity_source'] == 'encoder'
    assert meta['checkpoint_sha256'] == offline.PARENT_SHA
    assert not meta['native_observation_cfg']['noise'] and not meta['native_observation_cfg']['delay']
    assert len(meta['cases']) == 6
    observed=[]
    envs=[]
    for case in meta['cases']:
        assert case['acceleration'] in spec['accelerations']
        assert case['speed'] in card['future_final_common']['v_values']
        assert len(case['env_ids']) == 16
        observed.append((case['acceleration'],case['speed']))
        envs.extend(case['env_ids'])
        expected=[dict(start=float(row['start']),end=float(row['end']),
            speed=float(case['speed']*row['speed_multiplier'])) for row in spec['request_intervals']]
        actual=[{key:float(row[key]) for key in ('start','end','speed')} for row in case['segments']]
        assert actual == expected, 'Final requests differ from the preregistered card'
        assert all(row.get('acceleration',case['acceleration']) == case['acceleration'] for row in case['segments'])
    assert sorted(envs) == list(range(96))
    assert sorted(observed) == sorted((a,v) for a in spec['accelerations'] for v in card['future_final_common']['v_values'])


def input_only(folder,spec,card):
    meta=json.loads((folder/'encoder.json').read_text())
    check_protocol(meta,spec,card)
    assert json.loads((folder/'result.json').read_text())['verified']
    # Do not open target_com_m_s until both parent/candidate prediction files are sealed.
    with np.load(folder/'encoder.npz',allow_pickle=False) as data:
        history=data['native_history']
        frames=data['policy_observation']
        valid=data['policy_valid']
        original=data['encoder_m_s']
    assert history.shape == (3000,96,125) and frames.shape == (3000,96,25)
    assert valid.shape == (3000,96) and original.shape == (3000,96,3)
    assert np.isfinite(history).all() and valid[0].all()
    assert np.array_equal(history[valid],offline.rebuild_history(frames)[valid])
    return dict(meta=meta,history=history,valid=valid,original=original,
        files={str(path):offline.file_sha(path) for path in
            (folder/'encoder.json',folder/'encoder.npz',folder/'result.json',folder/'source.py')})


@torch.inference_mode()
def predict(encoder,contexts,device):
    result={}
    for run,context in contexts.items():
        history=context['history']
        flat=history.reshape(-1,125)
        values=np.empty((len(flat),3),dtype=np.float32)
        for begin in range(0,len(flat),8192):
            end=min(begin+8192,len(flat))
            batch=torch.from_numpy(np.array(flat[begin:end],copy=True)).to(device)
            values[begin:end]=(encoder(batch)/2.).cpu().numpy()
        assert np.isfinite(values).all()
        result[run]=values.reshape(*history.shape[:2],3)
    return result


def encoder_from(saved,device):
    state={key.removeprefix('encoder.'):value for key,value in saved['actor_state_dict'].items() if key.startswith('encoder.')}
    encoder=MLP(125,3,(128,64),'elu').to(device)
    encoder.load_state_dict(state,strict=True)
    encoder.eval().requires_grad_(False)
    return encoder,training_source.tree_digest(state)


def main(args):
    torch.set_num_threads(1)
    parent_path,candidate_path,card_path=(path.resolve() for path in (args.parent,args.candidate,args.execution_card))
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    assert offline.file_sha(parent_path) == offline.PARENT_SHA
    assert offline.file_sha(candidate_path) == CANDIDATE_SHA
    card=json.loads(card_path.read_text())
    offline.validate_gate_card(card)
    assert card['parent_checkpoint_sha256'] == offline.PARENT_SHA and card['authorized_encoder_updates'] == 1000
    receipt_path=candidate_path.parent/'checkpoint-1000.json'
    receipt=json.loads(receipt_path.read_text())
    assert receipt['update'] == 1000 and receipt['sha256'] == CANDIDATE_SHA and receipt['frozen_check_passed']
    development_path=candidate_path.parent/'terminal-development-gate.json'
    assert json.loads(development_path.read_text())['passed'], 'Final test requires the one terminal development gate to pass'
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    candidate=torch.load(candidate_path,map_location='cpu',weights_only=False)
    assert parent['iter'] == candidate['iter'] == 2999
    assert bitwise_equal(training_source.frozen_parts(parent),training_source.frozen_parts(candidate)), 'Non-encoder state changed'
    frozen={key:training_source.tree_digest(value) for key,value in training_source.frozen_parts(parent).items()}
    assert frozen == receipt['frozen_parts_sha256']
    assert offline.file_sha(offline.__file__) == offline.file_sha(candidate_path.parent/'offline-library-source.py')
    assert offline.file_sha(training_source.__file__) == offline.file_sha(candidate_path.parent/'trainer-source.py')
    folders={'seed48':args.capture48.resolve(),'seed49':args.capture49.resolve()}
    specs={f"seed{row['seed']}":row for row in card['future_final_runs']}
    assert set(folders) == set(specs)
    contexts={run:input_only(folder,specs[run],card) for run,folder in folders.items()}
    parent_encoder,parent_encoder_sha=encoder_from(parent,args.device)
    candidate_encoder,candidate_encoder_sha=encoder_from(candidate,args.device)
    assert candidate_encoder_sha == receipt['encoder_sha256']
    source_files={str(Path(path).resolve()):offline.file_sha(path) for path in
        (__file__,offline.__file__,training_source.__file__,inspect.getfile(MLP),card_path,receipt_path,development_path,parent_path,candidate_path)}
    lock=dict(stage='final_offline_test',single_candidate_update=1000,parent_sha256=offline.PARENT_SHA,
        candidate_sha256=CANDIDATE_SHA,candidate_encoder_sha256=candidate_encoder_sha,
        frozen_nonencoder_bitwise_equal=True,frozen_parts_sha256=frozen,
        source_files=source_files,captures={run:context['files'] for run,context in contexts.items()},
        gates=offline.OFFLINE_GATES,targets_opened=False,training_updates=0,control=False)
    write(output/'evaluation-lock.json',lock)
    shutil.copy2(__file__,output/'source.py')
    parent_predictions=predict(parent_encoder,contexts,args.device)
    candidate_predictions=predict(candidate_encoder,contexts,args.device)
    parity={}
    for run,context in contexts.items():
        residual=float(abs(parent_predictions[run][context['valid']]-context['original'][context['valid']]).max())
        parity[run]=dict(max_abs_m_s=residual,tolerance_m_s=1e-5,passed=residual <= 1e-5)
    assert all(value['passed'] for value in parity.values()), 'Parent output does not reproduce the new native capture'
    for name,values in (('parent',parent_predictions),('candidate',candidate_predictions)):
        np.savez_compressed(output/f'{name}-predictions.npz',**values)
    sealed={name:offline.file_sha(output/f'{name}-predictions.npz') for name in ('parent','candidate')}
    write(output/'predictions-sealed.json',dict(targets_opened=False,predictions_sha256=sealed,parent_parity=parity))
    # Full RunData/targets are opened only after immutable predictions above exist.
    split=offline.load_split(folders,name='final_test')
    write(output/'data-lock.json',split.lock())
    parent_score=offline.score_predictions(split,parent_predictions)
    candidate_score=offline.score_predictions(split,candidate_predictions,baseline=parent_score)
    write(output/'parent-metrics.json',parent_score)
    write(output/'candidate-metrics.json',candidate_score)
    gate=candidate_score['gate']
    per_run={}
    for run in folders:
        checks=[row for row in gate['rows'] if row['key'][0] == run]
        coverage={name:not any(row['run']==run for row in candidate_score['coverage'][name]) and
            not any(row['run']==run for row in parent_score['coverage'][name]) for name in
            ('failed_episodes','required_missing_phases','missing_complete_3s_parking_envs')}
        passed=all(coverage.values()) and all(row['passed'] is not False for row in checks)
        per_run[run]=dict(passed=passed,coverage=coverage,applicable_axis_segments=sum(row['status']!='not_applicable' for row in checks),
            failed_axis_segments=sum(row['passed'] is False for row in checks))
    assert training_source.tree_digest(training_source.cpu_state(parent_encoder)) == parent_encoder_sha
    assert training_source.tree_digest(training_source.cpu_state(candidate_encoder)) == candidate_encoder_sha
    assert all(offline.file_sha(Path(path)) == expected for path,expected in source_files.items())
    assert all(offline.file_sha(Path(path)) == expected for context in contexts.values() for path,expected in context['files'].items())
    assert all(offline.file_sha(output/f'{name}-predictions.npz') == expected for name,expected in sealed.items())
    split.verify_files()
    released=bool(gate['passed'] and all(row['passed'] for row in per_run.values()))
    result=dict(verified=True,stage='final_offline_test',parent_sha256=offline.PARENT_SHA,candidate_sha256=CANDIDATE_SHA,
        single_candidate_update=1000,training_updates=0,network_parameters_unchanged=True,
        frozen_nonencoder_bitwise_equal=True,parent_parity=parity,predictions_sealed_before_truth=True,
        label_alignment='Same current policy tick; no shifts',gate=gate,runs=per_run,
        closed_loop_release=released,
        next='Release only the preregistered frozen-controller paired simulation' if released else
            'Reject this terminal candidate for this fixed recipe; stop without retuning or recapturing this final test',
        scope='Offline estimator output only; no control behavior improvement or real-robot claim. One recipe failure is not universal method failure.',
        height='Backward low height remains independently unresolved; no height reward change.')
    write(output/'result.json',result)
    print(json.dumps(dict(verified=True,gate_passed=gate['passed'],runs=per_run,
        closed_loop_release=released,training_updates=0,control=False),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--capture48',type=Path,required=True)
    parser.add_argument('--capture49',type=Path,required=True)
    parser.add_argument('--execution-card',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    main(parser.parse_args())
