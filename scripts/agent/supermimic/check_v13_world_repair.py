"""Audit saved world-response repair budget, freeze proof and held-out gates."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def digest_state(state):
    h=hashlib.sha256()
    def add(value):
        if isinstance(value,torch.Tensor):h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value,dict):
            for k,v in value.items():h.update(str(k).encode());add(v)
        elif isinstance(value,(list,tuple)):
            for x in value:add(x)
        else:h.update(repr(value).encode())
    add(state);return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    root=args.experiment;result=json.loads((root/'result.json').read_text());recipe=json.loads((root/'recipe.json').read_text())
    candidate=torch.load(root/'candidate.pt',map_location='cpu',weights_only=False)
    parent=Path(candidate['parent_checkpoint']);assert sha(parent)==candidate['parent_sha256']
    saved=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    assert candidate['schema']=='v13-world-response-repair-candidate-v1'
    assert candidate['actual_additional_updates']==result['actual_world_updates']==recipe['additional_world_optimizer_updates']==300
    steps={int(v['step']) for v in candidate['opt_world']['state'].values()}
    assert steps=={saved['optimizer_update_counts']['world']+300}=={21356}
    assert candidate['optimizer_update_counts']==dict(world=21356,encoder=5200,actor=5200,critic=5200)
    for key,h in candidate['frozen_component_hashes'].items():assert digest_state(saved[key])==h
    for k,v in candidate['world'].items():
        assert torch.isfinite(v).all()
        if not k.startswith('net.'):assert torch.equal(v,saved['world'][k])
    assert any(not torch.equal(v,saved['world'][k]) for k,v in candidate['world'].items() if k.startswith('net.'))
    rows=[json.loads(line) for line in (root/'fit-metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,301))
    assert [r['world_updates'] for r in rows]==list(range(21057,21357))
    assert all(sha(Path(k))==v for k,v in recipe['training_inputs'].items())
    assert not recipe['validation_used_for_fitting'] and not recipe['sealed_benchmarks_used_for_fitting']
    assert result['candidate_sha256']==sha(root/'candidate.pt')
    for name,group in result['scores'].items():
        response=[r for r in group['response'] if r['epsilon']==.1]
        assert sum(r['same_direction'] for r in response)==result['aggregate'][name]['response']['correct']
        assert sum(r['eligible'] for r in response)==result['aggregate'][name]['response']['eligible']
    assert result['gates']['all_three_benchmarks_nonregressed']==all(b<=a for a,b in zip(result['benchmarks']['original'],result['benchmarks']['candidate']))
    assert result['accepted']==all(result['gates'].values())
    report=dict(all_audit_checks_passed=True,exact_world_updates=300,final_optimizer_counts=candidate['optimizer_update_counts'],frozen_components_verified=True,normalization_and_adapter_buffers_bitwise_equal=True,finite_model=True,training_inputs_hash_verified=True,candidate_accepted=result['accepted'],gates=result['gates'],source_checkpoint_sha256=sha(parent),candidate_sha256=sha(root/'candidate.pt'),audit_script_sha256=sha(Path(__file__)),scope='Saved-artifact audit, no optimization or simulation; candidate acceptance distinct from audit integrity.')
    output=root/'audit.json';assert not output.exists();output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
