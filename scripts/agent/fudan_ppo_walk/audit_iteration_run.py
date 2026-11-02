"""Audit completed iteration-budget checkpoints and source snapshots, without inference."""
import argparse,hashlib,json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    budget=json.loads((a.experiment/'budget.json').read_text());target=budget['iteration_budget']
    process_path=a.experiment/'process-result.json'
    process_exit_verified=process_path.exists()
    process=json.loads((process_path if process_exit_verified else a.experiment/'process-exit-observation.json').read_text())
    report=json.loads((a.run/'training_report.json').read_text())
    if process_exit_verified:assert process['returncode']==0
    else:assert process['training_process_returncode'] is None and process['training_pid_not_running']
    assert report['stopped_by']=='iteration_budget'
    assert report['iteration_budget']==report['iterations']==target
    assert report['transitions']==target*budget['num_envs']*budget['rollout_steps']
    assert report['primary_updates']==report['encoder_updates']==target*20
    assert not report['partial_update'] and report['checkpoint_sha256']==sha(a.run/'model_final.pt')
    initial=torch.load(a.run/'model_initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(a.run/'model_final.pt',map_location='cpu',weights_only=False)
    assert initial['iterations']==initial['transitions']==initial['primary_updates']==initial['encoder_updates']==0
    assert initial['initialization']==final['initialization']=='random_from_scratch_seed_1'
    assert not initial['optimizer_state_dict']['state'] and not initial['encoder_optimizer_state_dict']['state']
    assert final['reset_contract']==initial['reset_contract']
    assert final['iterations']==target and final['transitions']==report['transitions']
    assert not final['partial_update']
    optimizer_steps={}
    for name in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        values=[int(state['step']) for state in final[name]['state'].values()]
        assert values and set(values)=={target*20}
        optimizer_steps[name]=dict(parameters_with_state=len(values),minimum=min(values),maximum=max(values))
    assert all(torch.isfinite(value).all() for value in final['model_state_dict'].values())
    assert (final['model_state_dict']['std']>0).all()
    changed=any(not torch.equal(value,initial['model_state_dict'][name]) for name,value in final['model_state_dict'].items())
    assert changed
    checkpoints=[]
    for i in range(100,target+1,100):
        path=a.run/f'model_{i}.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['iterations']==i and saved['primary_updates']==saved['encoder_updates']==i*20
        assert saved['transitions']==i*budget['num_envs']*budget['rollout_steps']
        checkpoints.append(dict(iteration=i,path=str(path),sha256=sha(path)))
    manifest=json.loads((a.run/'source_sha256.json').read_text());checked=0
    prefix='mjlab_training/src/ct_mjlab/'
    for name,digest in manifest.items():
        if name.startswith(prefix):
            assert sha(a.run/'source/ct_mjlab'/name.removeprefix(prefix))==digest,name
            checked+=1
    baseline=json.loads((a.experiment/'baseline_sha256.json').read_text())
    changed_source=[name for name,digest in baseline.items() if manifest[name]!=digest]
    assert set(Path(name).name for name in changed_source)=={'__main__.py','runner.py'}
    result=dict(passed=True,scope='Counts, optimizer states, finite parameters and captured source; no behavioral validation',
        iterations=target,transitions=report['transitions'],primary_updates=target*20,encoder_updates=target*20,
        optimizer_steps=optimizer_steps,parameters_changed=bool(changed),parameter_finiteness=True,
        initial_checkpoint_sha256=sha(a.run/'model_initial.pt'),final_checkpoint_sha256=sha(a.run/'model_final.pt'),
        source_files_verified=checked,budget_only_source_changes=changed_source,checkpoints=checkpoints,
        process=process,process_exit_verified=process_exit_verified,training=report,physics_steps=0,optimizer_updates=0,network_forwards=0,
        behavior_accepted=False,cpu_launch_resolved=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoints','training')},indent=2))


if __name__=='__main__':main()
