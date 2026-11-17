"""Audit isolated speed experiments and the decision to preserve the active remote run."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.recording
    proof=json.loads((root/'equivalence-03/report.json').read_text())
    tensor=json.loads((root/'tensor-equivalence/report.json').read_text())
    assert proof['passed'] and tensor['passed'] and proof['optimizer_updates']==tensor['optimizer_updates']==0
    assert proof['all_contact_fields_bitwise_equal'] and proof['all_episode_fields_exact']
    fingerprints=json.loads((root/'candidate-fingerprints.json').read_text())
    for name,values in fingerprints.items():
        assert sha(root/'baseline-source'/name)==values['baseline_sha256']
        assert sha(root/'candidate-source'/name)==values['candidate_sha256']
        assert sha(Path(name))==values['baseline_sha256']
    samples={};reports={}
    source_sha=None
    for arm in ('baseline','candidate'):
        rows=[]
        for index in (1,2):
            name=f'raw-{arm}-{index:02d}';folder=root/name
            report=json.loads((folder/'report.json').read_text())
            restore=json.loads((folder/'runner/restore_report.json').read_text())
            assert report['passed'] and report['whole_only'] and report['optimizer_updates']==0
            assert report['all_weights_unchanged'] and report['optimizer_steps_unchanged']
            assert report['num_envs']==4096 and report['blocked_primary']==report['blocked_encoder']==60
            if source_sha is None:source_sha=restore['source']['sha256']
            assert restore['source']['sha256']==source_sha and restore['iterations']==200
            rows.extend(r['sample_s']+r['backward_no_adam_s']+r['log_s'] for r in report['rollouts'])
            reports[name]=report
        samples[arm]=rows
    means={key:statistics.mean(values) for key,values in samples.items()}
    profiled=json.loads((root/'baseline-profile-02/report.json').read_text())
    sample_total=sum(row['sample_s'] for row in profiled['rollouts'])
    share={key:profiled['stages'][key]['gpu_timeline_s']/sample_total for key in ('physics','contact_statistics')}
    before=(root/'remote-before.txt').read_text();after=(root/'remote-after.txt').read_text()
    assert '"pid": 7269' in before and '"pid": 7269' in after
    assert '"status": "training"' in after and '"budget": 3200' in after
    for name,values in fingerprints.items():
        assert values['baseline_sha256']+'  '+name in before
        assert values['baseline_sha256']+'  '+name in after
    result=dict(evidence_verified=True,switched=False,production_sources_restored=True,
        remote_pid=7269,remote_budget=3200,remote_sources_unchanged=True,
        optimizer_updates_during_validation=0,checkpoint_sha256=source_sha,
        same_input_contact_and_episode_equivalence=True,physics_contact_checks=proof['exact_contact_checks'],
        raw_samples_s=samples,mean_local_s=means,observed_local_reduction_fraction=1-means['candidate']/means['baseline'],
        sampling_phase_timeline_fraction=share,
        baseline_repeat_bitwise_equal=reports['raw-baseline-01']['tensors']==reports['raw-baseline-02']['tensors'],
        independent_baseline_candidate_bitwise_equal=reports['raw-baseline-01']['tensors']==reports['raw-candidate-01']['tensors'],
        l40_candidate_benchmarked=False,full_training_updates_compared=False,
        decision='Archive candidate only: modest local benefit, no target-GPU speed proof; preserve authorized active training.',
        scope='Local RTX4060 frozen sampling and full backward without Adam updates; no whole-training/L40 speedup or behavior claim.')
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
