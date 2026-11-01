"""Audit the bounded SM615-to-V13-v5 trial without optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from ct_mjlab.supermimic.learner import Learner,ALGORITHM_CONTRACT
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2)
    states={n:torch.load(args.run/f'{n}.pt',map_location='cpu',weights_only=False) for n in ('initial','prewarm','postwarm','final')}
    initial,prewarm,postwarm,final=[states[n] for n in ('initial','prewarm','postwarm','final')]
    config=final['config'];assert config['actor_updates']==300 and config['policy_warmup_updates']==4000
    expected={'initial':(0,0),'prewarm':(4000,0),'postwarm':(4004,1),'final':(5200,300)}
    for name,s in states.items():
        assert s['algorithm_contract']==ALGORITHM_CONTRACT
        w,a=expected[name];counts=Learner._saved_optimizer_counts(s)
        assert counts==s['optimizer_update_counts']==dict(world=w,encoder=a,actor=a,critic=a)
        assert s['world_updates']==w
        for net in ('actor','world','critic'):
            assert all(torch.isfinite(v).all() for v in s[net].values())
    parent=Path(initial['initialization']['parent_checkpoint'])
    assert sha(parent)==initial['initialization']['parent_sha256']
    teacher=torch.load(parent,map_location='cpu',weights_only=False)
    assert equal(initial['actor'],teacher['actor']) and equal(prewarm['actor'],initial['actor'])
    assert equal(prewarm['critic'],initial['critic'])
    assert not equal(prewarm['world'],initial['world']) and not equal(final['actor'],initial['actor'])
    assert initial['world']['net.0.weight'].shape[1]==129
    assert initial['world']['net.6.weight'].shape[0]==46
    restored_curriculum={}
    command=SimpleNamespace(cfg=command_cfg('walk'),load_checkpoint_state=lambda state:restored_curriculum.update(state))
    learner=Learner('cpu',actor_lr=3e-5);learner.load(args.run/'final.pt',command,expected_config=config)
    assert learner.actor.supermimic_action_clip==3.
    for name in ('actor','world','critic','opt_actor','opt_encoder','opt_world','opt_critic'):
        assert equal(getattr(learner,name).state_dict(),final[name])
    assert equal(restored_curriculum,final['curriculum'])
    assert torch.equal(torch.get_rng_state(),final['torch_rng'])
    cuda_rng_checked=torch.cuda.is_available()
    if cuda_rng_checked:
        assert equal(torch.cuda.get_rng_state_all(),final['cuda_rng'])
    manifest=json.loads((args.run/'source_sha256.json').read_text())
    root=Path(__file__).resolve().parents[3]
    for name,digest in manifest.items():
        assert sha(args.run/'snapshot/ct_mjlab'/name)==digest
        assert sha(root/'mjlab_training/src/ct_mjlab'/name)==digest
    metrics=[json.loads(line) for line in (args.run/'metrics.jsonl').read_text().splitlines()]
    assert max(r['actor_updates'] for r in metrics)==300
    assert all(r['actor_updates']<=300 for r in metrics)
    assert metrics[-1]['actor_updates']==300 and metrics[-1]['elapsed_s']<=config['seconds']+30
    report=dict(passed=True,counts=final['optimizer_update_counts'],transitions=final['transitions'],
        elapsed_learning_s=metrics[-1]['elapsed_s'],initial_actor_exact_SM615=True,
        warmup_actor_encoder_critic_unchanged=True,world_warmup_actual_updates=4000,
        network_optimizer_restore_equal=True,curriculum_restore_equal=True,cpu_rng_restore_equal=True,
        cuda_rng_restore_checked=cuda_rng_checked,action_clip=3.,source_files_unchanged=len(manifest),
        checkpoints={n:dict(sha256=sha(args.run/f'{n}.pt'),counts=s['optimizer_update_counts']) for n,s in states.items()},
        parent_sha256=sha(parent),scope='Training/count/restore audit only; behavior requires separate evaluation')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
