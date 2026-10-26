"""Pretrain an SMP diffusion model from qualified episodes; requires an explicit iteration budget."""
import argparse
from datetime import datetime
import json
import hashlib
from pathlib import Path
import shutil
import subprocess

import torch
from torch.utils.tensorboard import SummaryWriter

from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.prior import MotionPrior, checkpoint, load_prior
from ct_mjlab.utils.task_registry import LOG_ROOT,ROOT


def train(args):
    if args.iterations < 1 or args.batch_size < 1 or args.save_every < 1 or not 0 < args.lr < 1:
        raise ValueError('iterations/batch-size/save-every/lr must be positive')
    dataset = MotionDataset(args.dataset)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    prior = MotionPrior().to(args.device)
    first_iteration, saved = 0, None
    if args.resume:
        prior,saved = load_prior(args.resume,args.device)
        if saved['dataset']['fingerprint'] != dataset.fingerprint:
            raise ValueError('Resume dataset or split changed; full optimizer resume refused')
        first_iteration = saved['iteration']
    else:
        mean,std = dataset.normalization()
        prior.feature_mean.copy_(mean)
        prior.feature_std.copy_(std)
    optimizer = torch.optim.AdamW(prior.denoiser.parameters(),lr=args.lr)
    if saved is not None:
        if any(abs(group['lr']-args.lr)>1e-12 for group in saved['optimizer_state_dict']['param_groups']):
            raise ValueError('Resume learning rate differs; pass the original --lr for full resume')
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        generator.set_state(saved['generator_state'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    output = args.output or LOG_ROOT/'chuanliantui_smp_prior'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True,exist_ok=False)
    source_files = list((Path(__file__).resolve().parents[1]/'smp').glob('*.py')) + [Path(__file__).resolve(),ROOT/'mjlab_training/pyproject.toml']
    source_files += [ROOT/'third_party/mimickit_smp'/name for name in ('LICENSE','NOTICE.md')]
    source_files += list((ROOT/'mjlab_training/src/ct_mjlab/teacher').glob('*.py'))
    source_hashes = {}
    for source in source_files:
        relative = source.relative_to(ROOT)
        target = output/'snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    if (ROOT/'.git').exists():
        patch = subprocess.check_output(['git','diff','HEAD','--','mjlab_training'],cwd=ROOT,text=True)
    else:
        # 远程使用冻结源码包，保留打包端的Git来源，不要求创建或提交新仓库。
        patch = (ROOT/'migration/worktree.patch').read_text()
        shutil.copy2(ROOT/'migration/source.json',output/'snapshot/source.json')
    (output/'snapshot/worktree.patch').write_text(patch)
    description = dataset.description()
    (output/'dataset.json').write_text(json.dumps(description,indent=2)+'\n')
    (output/'arguments.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else
        [str(p) for p in v] if k=='dataset' else v for k,v in vars(args).items()},indent=2)+'\n')
    writer = SummaryWriter(str(output))
    print(f'SMP training output: {output}',flush=True)
    print(f'Additional iterations: {args.iterations}; groups split before windows; no PPO.',flush=True)
    validation_generator = torch.Generator().manual_seed(917)
    validation = dataset.sample('validation',min(args.batch_size,64),validation_generator).to(args.device)
    fixed_noise = torch.randn(validation.shape[0],300,generator=validation_generator).to(args.device)
    fixed_time = torch.arange(len(validation),device=args.device)%50
    prior.train()
    try:
        for iteration in range(first_iteration+1,first_iteration+args.iterations+1):
            windows = dataset.sample('train',args.batch_size,generator).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = prior.loss(windows)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite SMP training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.denoiser.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            prior.update_ema()
            writer.add_scalar('SMP/train_l1',float(loss.detach()),iteration)
            if iteration%args.save_every == 0 or iteration == first_iteration+args.iterations:
                with torch.no_grad():
                    validation_loss = prior.loss(validation,timestep=fixed_time,noise=fixed_noise)
                    normalized = prior.normalize(validation).flatten(1)
                    ema_prediction = prior.ema_denoiser(prior.add_noise(normalized,fixed_noise,fixed_time),fixed_time)
                    ema_loss = torch.nn.functional.l1_loss(ema_prediction,fixed_noise)
                writer.add_scalar('SMP/validation_l1',float(validation_loss),iteration)
                writer.add_scalar('SMP/validation_ema_l1',float(ema_loss),iteration)
                target = output/f'prior_{iteration}.pt'
                temporary = target.with_suffix('.tmp')
                state = checkpoint(prior,iteration=iteration,dataset=description,optimizer=optimizer,generator=generator)
                state['training_source_sha256'] = source_hashes
                torch.save(state,temporary)
                temporary.replace(target)
                writer.flush()
                print(f'iteration={iteration} train={float(loss.detach()):.6f} validation={float(validation_loss):.6f} saved={target}',flush=True)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,action='append',required=True)
    parser.add_argument('--iterations',type=int,required=True,help='Explicit additional optimization steps')
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--lr',type=float,default=.0002)
    parser.add_argument('--save-every',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--output',type=Path)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
