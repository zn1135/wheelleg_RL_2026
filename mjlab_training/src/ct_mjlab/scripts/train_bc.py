"""Behavior cloning of the existing actor; requires an explicit update budget."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

from ct_mjlab.bc import ActionDataset, FORMAT, WHEEL_FORMAT, LEGACY_SAMPLING, BRAKING_SAMPLING, PRESERVE_SAMPLING, build_actor, load_actor, expand_actor
from ct_mjlab.bc_preservation import migrate_wheel_actor,PreservationDataset
from ct_mjlab import reference_contract,wheel_angles
from ct_mjlab.utils.task_registry import ROOT


def train(args):
    if min(args.iterations, args.batch_size, args.save_every) < 1:
        raise ValueError('Explicit positive training budget, batch size and save interval required')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    # 新训练必须使用统一停车参考；旧数据只保留回放兼容。
    dataset = ActionDataset(args.dataset, args.corrections, action_sampling=args.action_sampling, require_reference=True)
    initialization = None
    migration=getattr(args,'migrate_wheel_angle',False)
    preserve_path=getattr(args,'preserve_skills',None)
    if migration!=(preserve_path is not None) or (migration and (args.init is None or args.action_sampling!=PRESERVE_SAMPLING)) or (
            args.action_sampling==PRESERVE_SAMPLING and not migration):
        raise ValueError('Preserved migration requires --init, --migrate-wheel-angle, --preserve-skills and preserved sampling together')
    if migration:
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends(allow_tf32=False)
    if args.init is not None:
        actor, old = load_actor(args.init, args.device)
        expanded=getattr(args,'expand_observations',False)
        source_dim=actor.mlp[0].in_features
        if expanded and migration:raise ValueError('Wheel migration and observation expansion are different operations')
        if not expanded and not migration and old['dataset']['observation_contract']!=dataset.description['observation_contract']:
            raise ValueError('BC initialization observation contracts differ, even with equal input dimensions')
        if migration:
            actor=migrate_wheel_actor(actor,old,dataset.description)
        elif expanded:
            if (dataset.observation_dim!=41 or source_dim not in (39,40) or
                    old['dataset']['xml_sha256']!=dataset.description['xml_sha256'] or
                    old['dataset']['control_sources']!=dataset.description['control_sources']):
                raise ValueError('Observation expansion requires compatible original39/40 motor control and XML')
            old_groups={s:{e['group_id'] for e in old['dataset']['splits'][s]} for s in ('train','validation')}
            if any(old_groups[s]!=set(dataset.description['group_split'][s]) for s in old_groups):
                raise ValueError('Observation expansion cannot change training/heldout pose groups')
            actor=expand_actor(actor,dataset.observation_dim,args.device)
        elif source_dim != dataset.observation_dim:
            raise ValueError('BC initialization observation dimensions differ; do not pad an old actor')
        if not expanded and not migration and old['dataset']['motion_fingerprint'] != dataset.description['motion_fingerprint']:
            raise ValueError('Correction initialization requires the same base demonstrations')
        initialization = dict(sha256=hashlib.sha256(args.init.read_bytes()).hexdigest(),
                              updates=old.get('total_actor_updates', old['iteration']),
                              **({'wheel_angle_migration':True,'zero_input_columns':[14,15],
                                  'source_observation_contract':old['dataset']['observation_contract'],
                                  'other_weights':'preserved'} if migration else {}),
                              **({'observation_expansion':[source_dim,dataset.observation_dim],
                                  'new_input_weights':'zero','old_learned_parameters':'preserved'} if expanded else {}))
    else:
        actor = build_actor(args.device, observation_dim=dataset.observation_dim)
    actor.train()
    preservation=PreservationDataset(preserve_path,dataset.description,initialization['sha256']) if migration else None
    optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=args.lr)
    # 只用训练集平衡六路动作的量级；网络仍直接输出原始PD动作，无新增部署缩放。
    target_scale = dataset.frames['train'][1].std(0, correction=0).clamp_min(.1).to(args.device)
    validation = dataset.sample('validation', 2048, torch.Generator().manual_seed(714))
    validation = tuple(v.to(args.device) for v in validation)
    args.output.mkdir(parents=True, exist_ok=False)
    description = dataset.description
    (args.output/'dataset.json').write_text(json.dumps(description, indent=2)+'\n')
    (args.output/'arguments.json').write_text(json.dumps({k: [str(p) for p in v] if k in ('dataset', 'corrections')
        else str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2)+'\n')
    files = list((ROOT/'mjlab_training/src/ct_mjlab').rglob('*.py'))
    source_hashes = {}
    for source in files:
        name = source.relative_to(ROOT)
        target = args.output/'snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(name)] = hashlib.sha256(source.read_bytes()).hexdigest()
    writer = SummaryWriter(str(args.output))
    print(f'BC output: {args.output}; explicit updates={args.iterations}; no PPO.', flush=True)
    try:
        for step in range(1, args.iterations+1):
            obs, target = (v.to(args.device) for v in dataset.sample('train', args.batch_size, generator))
            optimizer.zero_grad(set_to_none=True)
            prediction = actor.mlp(obs)
            bc_loss = ((prediction-target)/target_scale).square().mean()
            loss=bc_loss
            if preservation is not None:
                rise_loss,feedback_loss=preservation.loss(actor,generator,args.device)
                loss=loss+rise_loss+feedback_loss
                writer.add_scalar('BC/preserve_rise_scaled',float(rise_loss.detach()),step)
                writer.add_scalar('BC/preserve_feedback_scaled',float(feedback_loss.detach()),step)
                writer.add_scalar('BC/total_loss',float(loss.detach()),step)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite BC loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            writer.add_scalar('BC/train_mse_scaled', float(bc_loss.detach()), step)
            if step % args.save_every == 0 or step == args.iterations:
                with torch.no_grad():
                    error = actor.mlp(validation[0])-validation[1]
                    validation_loss = (error/target_scale).square().mean()
                writer.add_scalar('BC/validation_mse_scaled', float(validation_loss), step)
                for index, value in enumerate(error.abs().mean(0)):
                    writer.add_scalar(f'BC/validation_action_mae_{index}', float(value), step)
                model_format=(wheel_angles.PROFILES[description['observation_contract']]['model_format'] if description.get('wheel_angles') else
                              {39:FORMAT,40:WHEEL_FORMAT,41:reference_contract.FORMAT}[dataset.observation_dim])
                state = dict(format=model_format, iteration=step, actor_state_dict=actor.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), target_scale=target_scale,
                    dataset=description, generator_state=generator.get_state(), torch_rng=torch.get_rng_state(),
                    source_sha256=source_hashes, initialization=initialization,
                    total_actor_updates=step+(initialization['updates'] if initialization else 0))
                if preservation is not None:state['preservation']=preservation.description
                path = args.output/f'bc_{step}.pt'
                torch.save(state, path.with_suffix('.tmp'))
                path.with_suffix('.tmp').replace(path)
                writer.flush()
                print(f'iteration={step} train={float(loss.detach()):.6f} validation={float(validation_loss):.6f}', flush=True)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--corrections', type=Path, action='append', default=[])
    parser.add_argument('--action-sampling', choices=(LEGACY_SAMPLING, BRAKING_SAMPLING, PRESERVE_SAMPLING), default=LEGACY_SAMPLING)
    parser.add_argument('--init', type=Path, help='Initialize from a BC actor; starts a fresh optimizer with a new explicit budget')
    parser.add_argument('--expand-observations',action='store_true',help='Explicitly retain old39/40 weights and append zero inputs for shared-reference41')
    parser.add_argument('--migrate-wheel-angle',action='store_true',help='Explicit legacy39 to wheel39 migration; zero wheel input columns and preserve all other weights')
    parser.add_argument('--preserve-skills',type=Path,help='Checked train-only rise/local-feedback preservation manifest')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
