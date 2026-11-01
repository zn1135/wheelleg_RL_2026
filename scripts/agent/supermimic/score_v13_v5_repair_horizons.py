"""Fixed-scale1/8/24-control-step prediction checks on sealed unseen replay."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from ct_mjlab.supermimic.learner import Learner
from ct_mjlab.supermimic.networks import RunningNorm
from scripts.agent.supermimic.diagnose_v13_v5_response import sha
from scripts.agent.supermimic.check_v13_world_repair import digest_state


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','candidate','recording','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();assert not args.output.exists();torch.set_num_threads(2)
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    candidate=torch.load(args.candidate,map_location='cpu',weights_only=False)
    recording=torch.load(args.recording,map_location='cpu',weights_only=False)
    assert candidate['parent_sha256']==sha(args.checkpoint)==recording['checkpoint_hashes']['final']
    assert recording['seed']==candidate['recipe']['validation_seed']!=candidate['recipe']['training_seed']
    rows=[];scale=None
    for label,weights in [('parent',source['world']),('candidate',candidate['world'])]:
        learner=Learner('cpu');learner.world.load_state_dict(weights);learner.world.eval().requires_grad_(False)
        before=digest_state(learner.world.state_dict())
        if scale is None:scale=learner.world.delta_norm.std().clone()
        else:assert torch.equal(scale,learner.world.delta_norm.std())
        for length in (1,8,24):
            errors=[];positions=0;windows=0
            for s in recording['samples']:
                if s['anchor_windows'] is None:continue
                w={k:v[:,:length] for k,v in s['anchor_windows'].items()}
                assert w['action'].shape[1]==length
                loss,metrics=learner.world_loss(w)
                count=metrics['valid_world_positions'];errors.append(float(loss)*count);positions+=count;windows+=len(w['state'])
            rows.append(dict(model=label,horizon_steps=length,horizon_ms=length*10,
                normalized_mse=sum(errors)/positions,valid_positions=positions,replay_windows=windows))
        assert before==digest_state(learner.world.state_dict())
    changes={}
    for length in (1,8,24):
        before=next(r for r in rows if r['model']=='parent' and r['horizon_steps']==length)
        after=next(r for r in rows if r['model']=='candidate' and r['horizon_steps']==length)
        assert before['valid_positions']==after['valid_positions'] and before['replay_windows']==after['replay_windows']
        changes[str(length)]=dict(relative_change=after['normalized_mse']/before['normalized_mse']-1,
            nonregressed=after['normalized_mse']<=before['normalized_mse'])
    report=dict(rows=rows,changes=changes,all_horizons_nonregressed=all(x['nonregressed'] for x in changes.values()),
        fixed_scale=scale.tolist(),seed=recording['seed'],optimizer_updates=0,normalization_updates=0,
        checkpoint_sha256=sha(args.checkpoint),candidate_sha256=sha(args.candidate),recording_sha256=sha(args.recording),
        script_sha256=sha(Path(__file__)),scope='Same frozen real replay, free prediction without imagination clamps; no new control evaluation')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No updates')),patch.object(RunningNorm,'update',side_effect=AssertionError('No normalization updates')):
        main()
