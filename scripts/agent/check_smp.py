#!/usr/bin/env python3
"""Formal finite SMP checks; no training loop or optimizer step is run."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
from unittest.mock import patch
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ct_mjlab.smp.features import (FEATURE_DIM, FEATURE_SCHEMA, SAMPLE_DT, WINDOW,
                                   MotionHistory, inverse_rotate, motion_features)
from ct_mjlab.smp.dataset import MotionDataset, file_hash, behavior_name
from ct_mjlab.smp.prior import MotionPrior, PriorConfig, ErrorNormalizer, checkpoint, load_prior, SCORE_STEPS
from ct_mjlab.scripts.collect import qualify, command_grid

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/'third_party/mimickit_smp'


def original_functions():
    hashes = json.loads((REFERENCE/'reference_sha256.json').read_text())
    for name,digest in hashes.items():
        assert file_hash(REFERENCE/name) == digest, f'Changed upstream reference: {name}'
    tree = ast.parse((REFERENCE/'reference/tinymdm_model.py').read_text())
    cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == 'TinyMDMModel')
    functions = [node for node in cls.body if isinstance(node,ast.FunctionDef) and
                 node.name in ('ESM_SDS_loss','_get_epsilon','forward')]
    namespace = {'torch':torch}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned MimicKit methods>','exec'),namespace)
    return namespace,hashes


def feature_checks():
    n = 4
    root = torch.zeros(n,13)
    root[:,2],root[:,3] = .34,1.
    root[:,7:13] = torch.randn(n,6)
    motors,dq = torch.randn(n,6),torch.randn(n,6)
    wheel = root[:,:3,None].transpose(1,2)+torch.tensor([[.01,.18,-.27],[.01,-.18,-.27]])
    base = motion_features(root,motors,dq,wheel,torch.zeros(n))
    # Whole scene translation and heading rotation must not change features.
    angle = .73
    rotation = torch.tensor([[math.cos(angle),-math.sin(angle),0.],
                             [math.sin(angle),math.cos(angle),0.],[0.,0.,1.]])
    shifted = root.clone()
    offset = torch.tensor([12.,-4.,2.])
    shifted[:,:3] = root[:,:3]@rotation.T+offset
    shifted[:,3:7] = torch.tensor([math.cos(angle/2),0.,0.,math.sin(angle/2)])
    shifted[:,7:10] = root[:,7:10]@rotation.T
    shifted[:,10:13] = root[:,10:13]@rotation.T
    moved_wheels = wheel@rotation.T+offset
    moved = motion_features(shifted,motors,dq,moved_wheels,torch.full((n,),2.))
    torch.testing.assert_close(base,moved,atol=2e-6,rtol=2e-6)
    motors[:,[2,5]] += 1234  # 累计轮角完全排除。
    torch.testing.assert_close(base,motion_features(root,motors,dq,wheel,torch.zeros(n)),atol=0,rtol=0)
    motors[:,[0,1,3,4]] += 2*torch.pi
    torch.testing.assert_close(base,motion_features(root,motors,dq,wheel,torch.zeros(n)),atol=2e-6,rtol=2e-6)
    history = MotionHistory(n,'cpu')
    for tick in range(WINDOW-1):
        _,valid = history.append(base+tick)
        assert not valid.any()
    frames,valid = history.append(base+9)
    assert valid.all()
    torch.testing.assert_close(frames[:,0],base)
    other = history.frames[[0,2]].clone()
    history.reset(torch.tensor([1,3]))
    assert not history.count[[1,3]].any()
    torch.testing.assert_close(history.frames[[0,2]],other,atol=0,rtol=0)
    history.append(base)
    assert history.count.tolist() == [10,1,10,1]
    damaged = base.clone()
    damaged[1,0] = float('nan')
    _,valid = history.append(damaged)
    assert not valid[1] and not history.frames[1].any()
    return dict(dim=FEATURE_DIM,window=WINDOW,heading_translation_and_wheel_spin_invariant=True,
                partial_reset_and_warmup=True)


def fixtures(directory):
    records = []
    for group in range(6):
        for env in range(2):
            frames = 40
            root = torch.zeros(frames,13)
            root[:,2],root[:,3] = .3+group*.01,1.
            behavior = ('standing','standup','forward')[group//2]
            phase = np.full(frames,3,dtype=np.int64)
            if behavior == 'standup':
                phase[:12] = 2
                root[:12,2] = .1
            root[:,7] = group+env*.1
            q = torch.arange(frames)[:,None].repeat(1,6).float()*.001+group
            dq = torch.zeros_like(q)
            wheels = root[:,:3,None].transpose(1,2)+torch.tensor([[0.,.18,-.25],[0.,-.18,-.25]])
            ground = torch.zeros(frames)
            f = motion_features(root,q,dq,wheels,ground)
            row = dict(episode_id=f'group{group}-env{env}',group_id=f'group{group}',behavior=behavior,
                       accepted=True,rejection_reasons=[],path=f'{group}-{env}.npz')
            fixed_command = {'vx':.5 if behavior == 'forward' else 0.,'yaw':0.,'leg_length':.3}
            meta = dict(row,xml_sha256='synthetic_xml',feature_schema=FEATURE_SCHEMA,synthetic_test_only=True,
                        initial='ground' if behavior == 'standup' else 'standing',
                        schedule=[dict(time=0.,**fixed_command)])
            target = directory/row['path']
            np.savez_compressed(target,time=np.arange(frames)*SAMPLE_DT,root=root.numpy(),motor_q=q.numpy(),
                motor_dq=dq.numpy(),wheel_position_w=wheels.numpy(),ground_height=ground.numpy(),features=f.numpy(),
                phase=phase,command=np.tile([fixed_command[k] for k in ('vx','yaw','leg_length')],(frames,1)),
                metadata=np.asarray(json.dumps(meta)))
            row['sha256'] = file_hash(target)
            records.append(row)
    manifest = directory/'dataset.json'
    manifest.write_text(json.dumps(dict(schema='chuanliantui_smp_episodes_v1',feature_schema=FEATURE_SCHEMA,
        feature_dim=FEATURE_DIM,sample_dt=SAMPLE_DT,xml_sha256='synthetic_xml',episodes=records)))
    return manifest


def dataset_checks(directory):
    path = fixtures(directory)
    data = MotionDataset([path])
    train_groups = {e['group_id'] for e in data.splits['train']}
    val_groups = {e['group_id'] for e in data.splits['validation']}
    assert not train_groups&val_groups
    generator = torch.Generator().manual_seed(7)
    windows = data.sample('train',512,generator)
    # Within each episode, this channel is constant. A boundary-crossing window changes it.
    assert (windows[:,:,4].amax(1) == windows[:,:,4].amin(1)).all()
    starts = data._starts['train']
    assert len(starts) == 6*(40-WINDOW+1)
    assert set(data._class_starts['train']) == {'standing','standup','forward'}
    assert set(data._class_starts['validation']) == {'standing','standup','forward'}
    assert .25 < float(((windows[:,0,4] >= 2)&(windows[:,0,4] < 4)).float().mean()) < .42
    assert .25 < float((windows[:,0,4] >= 4).float().mean()) < .42
    before = data.normalization()
    for episode in data.splits['validation']:
        episode['features'].fill_(1000)
    after = data.normalization()
    for a,b in zip(before,after):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    try:
        MotionDataset([path,path])
    except ValueError as exc:
        assert 'Duplicate episode' in str(exc)
    else:
        raise AssertionError('duplicate episode accepted')
    # Tampering, rejected episodes and incompatible data must be refused.
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    changed['episodes'][0]['accepted'] = False
    path.write_text(json.dumps(changed))
    try:
        MotionDataset([path])
    except ValueError:
        pass
    else:
        raise AssertionError('rejected episode accepted')
    path.write_text(json.dumps(original))
    path.write_text(json.dumps(dict(original,episodes=original['episodes'][::-1])))
    reordered = MotionDataset([path])
    path.write_text(json.dumps(original))
    canonical = MotionDataset([path])
    assert canonical.fingerprint == reordered.fingerprint
    torch.testing.assert_close(canonical.sample('train',64,torch.Generator().manual_seed(19)),
                               reordered.sample('train',64,torch.Generator().manual_seed(19)),atol=0,rtol=0)
    missing_group = dict(original,episodes=[r for r in original['episodes'] if r['group_id'] != 'group5'])
    path.write_text(json.dumps(missing_group))
    try:
        MotionDataset([path])
    except ValueError as exc:
        assert 'each behavior' in str(exc)
    else:
        raise AssertionError('behavior without a validation seed accepted')
    path.write_text(json.dumps(original))
    grid_path = directory/'commands.json'
    rows = [{'vx':.5,'yaw':0.,'leg_length':.3},{'vx':-.5,'yaw':0.,'leg_length':.3}]
    grid_path.write_text(json.dumps(rows))
    grid = command_grid(SimpleNamespace(commands=grid_path,sequence=None,num_envs=4))
    assert grid == rows+rows
    assert [behavior_name('standing',[row]) for row in grid] == ['forward','backward']*2
    assert behavior_name('ground',[rows[0]]) == 'standup'
    gated_records = []
    for record in original['episodes']:
        changed = dict(record)
        if record['behavior'] == 'standup':
            with np.load(directory/record['path']) as stored:
                arrays = {key: stored[key].copy() for key in stored.files}
            metadata = json.loads(str(arrays['metadata']))
            metadata['ground_start_hold_s'] = .2
            arrays['metadata'] = np.asarray(json.dumps(metadata))
            changed['path'] = 'gated-'+record['path']
            np.savez_compressed(directory/changed['path'], **arrays)
            changed['sha256'] = file_hash(directory/changed['path'])
        gated_records.append(changed)
    gated_manifest = directory/'gated.json'
    gated_manifest.write_text(json.dumps(dict(original, ground_start_hold_s=.2, episodes=gated_records)))
    gated = MotionDataset([gated_manifest])
    assert gated.description()['ground_start_hold_s'] == .2 and gated.fingerprint != canonical.fingerprint
    old_manifest, new_manifest = directory/'old-ground.json', directory/'new-ground.json'
    old_manifest.write_text(json.dumps(dict(original, episodes=[
        record for record in original['episodes'] if record['group_id'] == 'group2'])))
    new_manifest.write_text(json.dumps(dict(original, ground_start_hold_s=.2, episodes=[
        record for record in gated_records if record['group_id'] == 'group3'])))
    try:
        MotionDataset([old_manifest, new_manifest])
    except ValueError as exc:
        assert 'different startup holds' in str(exc)
    else:
        raise AssertionError('Mixed immediate and delayed standup demonstrations accepted')
    return dict(groups=6,episodes=12,train_groups=sorted(train_groups),validation_groups=sorted(val_groups),
                window_boundaries_and_train_only_statistics=True,duplicate_and_rejected_refused=True,
                each_behavior_in_both_splits=True,balanced_rise_and_movement_sampling=True,
                missing_behavior_group_refused=True,fixed_command_grid=True,manifest_order_invariant=True,
                ground_start_hold_recorded_and_mixed_holds_refused=True)


def score_checks(directory):
    functions,hashes = original_functions()
    prior = MotionPrior()
    windows = torch.randn(4,WINDOW,FEATURE_DIM)
    noise = torch.randn(3,len(windows),WINDOW*FEATURE_DIM)
    x = prior.normalize(windows).flatten(1)
    scheduler = SimpleNamespace(alphas_cumprod=prior.alpha_cumprod,
        config=SimpleNamespace(prediction_type='epsilon'))

    def reference_noise(samples,noise,timesteps):
        alpha = scheduler.alphas_cumprod[timesteps][:,None]
        return samples*torch.sqrt(alpha)+noise*torch.sqrt(1-alpha)

    def ddim_step(pred,t,noised):
        alpha = scheduler.alphas_cumprod[t]
        return SimpleNamespace(prev_sample=torch.zeros_like(noised),
            pred_original_sample=(noised-torch.sqrt(1-alpha)*pred)/torch.sqrt(alpha))

    scheduler.add_noise = lambda original_samples,noise,timesteps: reference_noise(original_samples,noise,timesteps)
    adapter = SimpleNamespace(model_ema=True,diffusion_scheduler=scheduler,ddim_scheduler=SimpleNamespace(step=ddim_step),
        _get_sampling_denoiser=lambda **kwargs: prior.ema_denoiser)
    adapter._get_epsilon = MethodType(functions['_get_epsilon'],adapter)
    draws = iter(noise)
    with patch('torch.randn_like',side_effect=lambda value: next(draws)):
        expected = functions['ESM_SDS_loss'](adapter,x,t_lst=SCORE_STEPS)
    actual = prior.score_errors(windows,noise=noise)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=2e-6)
    timestep = torch.tensor([0,8,22,49])
    train_adapter = SimpleNamespace(diffusion_scheduler=scheduler,dmodel=prior.denoiser,loss_type='l1')
    with patch('torch.randn_like',return_value=noise[0]):
        expected_loss = functions['forward'](train_adapter,x,timesteps=timestep)
    loss = prior.loss(windows,timestep=timestep,noise=noise[0])
    torch.testing.assert_close(loss,expected_loss,atol=1e-7,rtol=1e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in prior.denoiser.parameters())
    assert any(p.grad.abs().sum()>0 for p in prior.denoiser.parameters())
    assert not any(p.grad is not None for p in prior.ema_denoiser.parameters())
    # Actual upstream EMA, driven by deterministic manual weights (no optimization).
    namespace = {'torch':torch,'nn':torch.nn,'deepcopy':__import__('copy').deepcopy}
    exec(compile((REFERENCE/'reference/EMA.py').read_text(),'<pinned EMA>','exec'),namespace)
    ref_ema = namespace['EMA'](prior.denoiser,beta=.995,update_every=10,update_after_step=5000)
    prior.ema_step.fill_(5000)
    ref_ema.step.fill_(5000)
    for tick in range(12):
        with torch.no_grad():
            for p in prior.denoiser.parameters():
                p.add_(.0001)
        prior.update_ema()
        ref_ema.update()
    for a,b in zip(prior.ema_denoiser.parameters(),ref_ema.ema_model.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    normalizer = prior.error_normalizer
    normalizer.record(actual)
    normalizer.update()
    torch.testing.assert_close(normalizer.mean_abs,actual.mean(0))
    normalizer_tree = ast.parse((REFERENCE/'reference/diff_normalizer.py').read_text())
    normalizer_class = next(n for n in normalizer_tree.body if isinstance(n,ast.ClassDef))
    namespace = {'torch':torch,'np':np,'mp_util':SimpleNamespace(reduce_sum=lambda x:x,
                  reduce_inplace_sum=lambda x:None),'Logger':SimpleNamespace(print=print)}
    exec(compile(ast.Module(body=[normalizer_class],type_ignores=[]),'<pinned DiffNormalizer>','exec'),namespace)
    ref_normalizer = namespace['DiffNormalizer']([3],device='cpu')
    ref_normalizer.record(actual)
    ref_normalizer.update()
    torch.testing.assert_close(normalizer(actual),ref_normalizer.normalize(actual),atol=1e-7,rtol=1e-6)
    reward = prior.score(windows,noise=noise)
    # We changed the weights for the EMA check; use current errors in the independent reward formula.
    current_error = prior.score_errors(windows,noise=noise)
    torch.testing.assert_close(reward,torch.exp(-6*(current_error/actual.mean(0)).mean(-1)))
    optimizer = torch.optim.AdamW(prior.denoiser.parameters(),lr=.0002)
    first_parameter = next(prior.denoiser.parameters())
    optimizer.state[first_parameter] = dict(step=torch.tensor(5.),exp_avg=torch.full_like(first_parameter,.01),
                                            exp_avg_sq=torch.full_like(first_parameter,.001))
    generator = torch.Generator().manual_seed(51)
    target = directory/'finite-fixture.pt'
    normalizer.record(current_error[:1])  # 尚未合并的统计也必须随检查点恢复。
    torch.save(checkpoint(prior,iteration=0,dataset={'fingerprint':'synthetic'},
                          optimizer=optimizer,generator=generator),target)
    restored,saved = load_prior(target,require_trained=False)
    restored_optimizer = torch.optim.AdamW(restored.denoiser.parameters(),lr=.0002)
    restored_optimizer.load_state_dict(saved['optimizer_state_dict'])
    restored_state = restored_optimizer.state[next(restored.denoiser.parameters())]
    for key,value in optimizer.state[first_parameter].items():
        torch.testing.assert_close(restored_state[key],value,atol=0,rtol=0)
    torch.testing.assert_close(restored.score(windows,noise=noise),reward,atol=0,rtol=0)
    normalizer.update()
    restored.error_normalizer.update()
    for name,value in normalizer.state_dict().items():
        torch.testing.assert_close(restored.error_normalizer.state_dict()[name],value,atol=0,rtol=0)
    try:
        load_prior(target)
    except ValueError:
        pass
    else:
        raise AssertionError('untrained fixture accepted as a trained prior')
    expected_draw = torch.randint(100,(20,),generator=generator)
    generator.set_state(saved['generator_state'])
    torch.testing.assert_close(torch.randint(100,(20,),generator=generator),expected_draw)
    restored.freeze()
    assert not any(p.requires_grad for p in restored.parameters())
    return dict(upstream_reference_sha256=hashes,score_max_abs=float((actual-expected).abs().max()),
                train_loss_matches=True,finite_nonzero_gradients=True,optimizer_steps=0,
                ema_matches_upstream=True,error_normalizer_matches_upstream=True,
                checkpoint_score_optimizer_and_sampler_match=True,pending_statistics_restored=True,
                untrained_checkpoint_refused=True,frozen_parameters=True)


def network_checks():
    from ct_mjlab.smp.dit import TinyMotionDiT, SwiGLU
    from torch import nn
    # Original MimicKit network/block code, with independent torch MHA instead of diffusers Attention.
    class ReferenceAttention(nn.Module):
        def __init__(self, query_dim, heads, dim_head, **kwargs):
            super().__init__()
            self.heads = heads
            self.to_q,self.to_k,self.to_v = (nn.Linear(query_dim,query_dim,bias=False) for _ in range(3))
            self.to_out = nn.ModuleList((nn.Linear(query_dim,query_dim),nn.Dropout(0.)))

        def forward(self, x, **kwargs):
            from torch.nn.functional import multi_head_attention_forward
            seq = x.transpose(0,1)
            value,_ = multi_head_attention_forward(seq,seq,seq,x.shape[-1],self.heads,
                None,None,None,None,False,0.,self.to_out[0].weight,self.to_out[0].bias,
                training=False,need_weights=True,use_separate_proj_weight=True,
                q_proj_weight=self.to_q.weight,k_proj_weight=self.to_k.weight,v_proj_weight=self.to_v.weight)
            return value.transpose(0,1)

    class ReferenceTimesteps(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, timestep):
            frequencies = torch.from_numpy(np.exp(-np.log(10000)*np.arange(128,dtype=np.float32)/128)).float()
            values = timestep.float()[:,None]*frequencies
            return torch.cat((values.cos(),values.sin()),-1)

    class ReferenceEmbedding(nn.Module):
        def __init__(self, in_channels, time_embed_dim):
            super().__init__()
            self.linear_1,self.linear_2 = nn.Linear(in_channels,time_embed_dim),nn.Linear(time_embed_dim,time_embed_dim)
        def forward(self, x):
            return self.linear_2(torch.nn.functional.silu(self.linear_1(x)))

    namespace = dict(torch=torch,nn=nn,math=math,Any=Any,Dict=Dict,List=List,Optional=Optional,Tuple=Tuple,
        Attention=ReferenceAttention,Timesteps=ReferenceTimesteps,TimestepEmbedding=ReferenceEmbedding,
        SwiGLU=lambda dim_in,dim_out,bias=True:SwiGLU(dim_in),ConfigMixin=type('ConfigMixin',(),{}))
    for name in ('modules.py','arch.py'):
        tree = ast.parse((REFERENCE/'reference'/name).read_text())
        classes = [node for node in tree.body if isinstance(node,ast.ClassDef)]
        exec(compile(ast.Module(body=classes,type_ignores=[]),'<pinned '+name+'>','exec'),namespace)
    reference = namespace['TinyStableMotionDiTModel'](in_channels=30,out_channels=30,num_layers=2)
    port = TinyMotionDiT()
    mapping = {}
    for name,value in port.state_dict().items():
        name = name.replace('time_embedding.','adaln_single.emb.timestep_embedder.')
        name = name.replace('time_linear.','adaln_single.linear.')
        name = name.replace('position_encoding','sequence_pos_encoder.pe')
        mapping[name] = value
    missing,extra = reference.load_state_dict(mapping,strict=False)
    assert missing == ['scale_shift_table'] and not extra
    # Same fixed weights and inputs: all actual upstream block/wiring paths execute.
    x,t = torch.randn(4,300),torch.tensor([0,8,22,49])
    actual,expected = port(x,t),reference(x,t)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    result = dict(upstream_dit_max_abs=float((actual-expected).abs().max().detach()),
                  independent_attention='torch multi_head_attention_forward',
                  unused_output_parameter_removed=True)
    if torch.cuda.is_available():
        noise = torch.randn(3,4,300)
        prior = MotionPrior().eval()
        cpu_score = prior.score(x.reshape(4,10,30),noise=noise)
        prior = prior.to('cuda:0')
        gpu_score = prior.score(x.reshape(4,10,30).cuda(),noise=noise.cuda()).cpu()
        torch.testing.assert_close(cpu_score,gpu_score,atol=1e-7,rtol=2e-4)
        result['cuda_score_max_abs'] = float((cpu_score-gpu_score).abs().max())
    return result


def qualification_checks():
    result = dict(final_balance_v31=True,phase='balance',ever_failed=False,
                  mean_abs_vx_error=.1,mean_abs_yaw_error=.1,mean_abs_height_error=.01)
    assert qualify(result,tracking_tilt=0.,tracking_contact=False,segments=1)[0]
    result['mean_abs_vx_error'] = .56
    assert not qualify(result,tracking_tilt=0.,tracking_contact=False,segments=1)[0]
    result['mean_abs_vx_error'] = .1
    assert not qualify(result,tracking_tilt=0.,tracking_contact=False,segments=2)[0]
    result['mean_abs_yaw_error'] = float('nan')
    assert not qualify(result,tracking_tilt=0.,tracking_contact=False,segments=1)[0]
    return dict(rolling_balance_alone_refused=True,unvalidated_sequence_refused=True,
                nonfinite_tracking_refused=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,action='append')
    parser.add_argument('--require-behaviors',help='Comma-separated motion classes required in both splits')
    args = parser.parse_args()
    if args.require_behaviors and not args.dataset:
        parser.error('--require-behaviors needs --dataset')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    sources = [Path(__file__),ROOT/'mjlab_training/src/ct_mjlab/cli.py']
    sources += sorted((ROOT/'mjlab_training/src/ct_mjlab/smp').glob('*.py'))
    sources += sorted((ROOT/'mjlab_training/src/ct_mjlab/teacher').glob('*.py'))
    sources += [ROOT/'mjlab_training/src/ct_mjlab/scripts'/name for name in ('collect.py','train_prior.py')]
    report = dict(passed=False,training_started=False,
                  source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sources})
    try:
        report['features'] = feature_checks()
        report['qualification'] = qualification_checks()
        with tempfile.TemporaryDirectory(prefix='ct-smp-check-') as tmp:
            directory = Path(tmp)
            report['dataset'] = dataset_checks(directory)
            report['score'] = score_checks(directory)
            report['network'] = network_checks()
        if args.dataset:
            data = MotionDataset(args.dataset)
            required = set(args.require_behaviors.split(',')) if args.require_behaviors else set()
            for split,classes in data._class_starts.items():
                if not required <= classes.keys():
                    raise ValueError(f'{split} missing requested motion classes: {sorted(required-classes.keys())}')
            prior = MotionPrior()
            mean,std = data.normalization()
            prior.feature_mean.copy_(mean)
            prior.feature_std.copy_(std)
            batch = data.sample('train',16,torch.Generator().manual_seed(81))
            loss = prior.loss(batch)
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
                                                for p in prior.denoiser.parameters())
            scores = prior.score(batch)
            assert torch.isfinite(scores).all()
            report['real_dataset'] = dict(data.description(),
                frame_counts={split:len(frames) for split,frames in data._frames.items()},
                window_counts={split:len(starts) for split,starts in data._starts.items()},
                required_behaviors=sorted(required),
                finite_loss_and_gradients=True,finite_scores=True,optimizer_steps=0)
            if data.ground_start_hold_s:
                early_frames, first_active = 0, []
                for episode in data.episodes:
                    if episode['behavior'] != 'standup':
                        continue
                    with np.load(episode['path']) as trace:
                        early = trace['time'] <= data.ground_start_hold_s+1e-8
                        assert early.any() and not trace['control_active'][early].any()
                        assert not trace['torque'][early].any() and not trace['actuator_torque'][early].any()
                        assert trace['control_active'][~early].all()
                        first = np.flatnonzero(np.abs(trace['torque']).max(-1) > 1e-3)
                        assert len(first) and trace['time'][first[0]] > data.ground_start_hold_s
                        early_frames += int(early.sum())
                        first_active.append(float(trace['time'][first[0]]))
                report['real_dataset']['startup_motor_hold'] = dict(
                    verified_early_frames=early_frames, requested_and_actual_motor_torques_zero=True,
                    first_nonzero_recorded_times_s=first_active)
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
