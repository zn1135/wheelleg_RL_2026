"""Resume the interrupted3018 learning state for the remaining982 authorized iterations."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from scripts.agent.fudan_ppo_walk import resume_pd200_random_from3000 as base

ROOT=base.ROOT
PARENT=ROOT/'artifacts/fudan_ppo_walk/20261115-pd200-full-random-force3-push-resume3000-4000-01'
SOURCE_RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261115_pd200_full_random_ground04to12_force3_push_resume3000_to4000_01'
MODEL=SOURCE_RUN/'model_final.pt'
LOG=ROOT/'logs/mjlab/fudan_ppo_walk/20261115_pd200_full_random_force3_push_resume3018_to4000_01'
MODULE='scripts.agent.fudan_ppo_walk.resume_pd200_remaining'


def prepare(output):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    source=torch.load(MODEL,map_location='cpu',weights_only=False)
    assert base.sha(MODEL)=='d266c5d4091a99d6e5f92c79f689d6185ddd332fde38ddf31f97b7c9cf3fa2c2'
    assert source['iterations']==3018 and not source['partial_update']
    assert source['primary_updates']==source['encoder_updates']==3018*20
    # 中断发生在下一轮采样内：学习更新完整，额外采样计数如实保留，不伪造完整边界。
    abandoned=source['transitions']-3018*4096*48
    assert abandoned==184320 and 'self.env.step(actions)' in (PARENT/'train.log').read_text()
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in source[key]['state'].values()}=={60360}
        assert all(torch.isfinite(value).all() for state in source[key]['state'].values()
                   for value in state.values() if isinstance(value,torch.Tensor))
    assert all(torch.isfinite(v).all() for v in source['model_state_dict'].values())
    events=EventAccumulator(str(SOURCE_RUN),size_guidance={'scalars':0}).Reload()
    assert [v.step for v in events.Scalars('Train/reward')]==list(range(3001,3019))
    manifest=json.loads((SOURCE_RUN/'source_sha256.json').read_text())
    for name,digest in manifest.items():
        if name.startswith(('mjlab_training/src/ct_mjlab/','resources/')):assert base.sha(ROOT/name)==digest,name
    assert not LOG.exists()
    budget=json.loads((PARENT/'budget.json').read_text())
    assert budget['actual_config']==json.loads((SOURCE_RUN/'effective-training-config.json').read_text())
    assert source['disturbances']==budget['disturbances']
    budget.update(authorization='User requested continuing the previously stopped run; use remaining982 of the original1000, stop at4000',
        checkpoint=str(MODEL),checkpoint_sha256=base.sha(MODEL),start_iteration=3018,target_iteration=4000,
        iteration_budget=982,additional_iterations=982,log_dir=str(LOG),only_active_config_changes={},
        source_config_sha256=base.sha(SOURCE_RUN/'effective-training-config.json'),
        source_recorded_transitions=source['transitions'],source_completed_update_transitions=3018*4096*48,
        inherited_abandoned_rollout_transitions=abandoned,created_unix=time.time())
    output.mkdir(parents=True,exist_ok=False);base.write(output/'budget.json',budget)
    for folder in ('precheck-force3-push','friction-check'):
        (output/folder).mkdir();shutil.copy2(PARENT/folder/'report.json',output/folder/'report.json')
    for name in ('precheck-force3-push-external.json','precheck-force3-push-contact-friction.json'):
        shutil.copy2(PARENT/name,output/name)
    shutil.copy2(__file__,output/'resume-launcher.py');shutil.copy2(base.__file__,output/'launcher.py')
    shutil.copy2(SOURCE_RUN/'effective-training-config.json',output/'source-effective-config.json')
    base.write(output/'resume-source-audit.json',dict(passed=True,iteration=3018,both_adam_steps=60360,
        source_checkpoint_sha256=base.sha(MODEL),interruption_stage='rollout_before_next_update',
        inherited_abandoned_rollout_transitions=abandoned,source_counters_unchanged=True,
        runtime_physics_sources_exact=True,precheck_reused_from=str(PARENT),active_config_changes={}))
    base.write(output/'runtime-source-sha256.json',{name:digest for name,digest in manifest.items()
        if name.startswith(('mjlab_training/src/ct_mjlab/','resources/'))})
    base.write(output/'launcher-source-sha256.json',{str(Path(file).relative_to(ROOT)):base.sha(file)
        for file in (__file__,base.__file__,ROOT/'scripts/agent/fudan_ppo_walk/check_ground_friction.py')})
    (output/'training-workspace.patch').write_bytes(subprocess.check_output(['git','diff','--binary','--',
        'mjlab_training/','scripts/agent/fudan_ppo_walk/resume_pd200_random_from3000.py'],cwd=ROOT))
    base.write(output/'status.json',dict(status='prepared',remaining_authorized_iterations=982))


def audit(output):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    budget=json.loads((output/'budget.json').read_text())
    source=torch.load(MODEL,map_location='cpu',weights_only=False)
    initial=torch.load(LOG/'model_initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(LOG/'model_final.pt',map_location='cpu',weights_only=False)
    report=json.loads((LOG/'training_report.json').read_text())
    assert json.loads((output/'process-result.json').read_text())['returncode']==0
    assert json.loads((output/'initialization-audit.json').read_text())['passed']
    for key in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng',
                'transitions','primary_updates','encoder_updates','speed_limit','policy_steps'):
        assert base.same(initial[key],source[key]),key
    assert initial['iterations']==3018 and final['iterations']==4000 and not final['partial_update']
    assert report['stopped_by']=='iteration_budget' and report['iterations_added']==report['iteration_budget']==982
    assert final['transitions']-source['transitions']==982*4096*48
    assert final['primary_updates']==final['encoder_updates']==80000
    for key in ('control','control_timing','height_reward','disturbances','physical_wheel_model'):
        assert initial[key]==final[key]==source[key],key
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in final[key]['state'].values()}=={80000}
    assert all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
    events=EventAccumulator(str(LOG),size_guidance={'scalars':0}).Reload()
    for tag in ['Train/reward',*['Rewards/'+name for name in final['reward_names']]]:
        assert [v.step for v in events.Scalars(tag)]==list(range(3019,4001)),tag
    for manifest in ('runtime-source-sha256.json','launcher-source-sha256.json'):
        for name,digest in json.loads((output/manifest).read_text()).items():assert base.sha(ROOT/name)==digest,name
    for name,digest in json.loads((LOG/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert base.sha(LOG/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest,name
    base.write(output/'training-audit.json',dict(passed=True,source_iteration=3018,final_iteration=4000,
        additional_iterations=982,original_budget_completed_iterations=18+982,
        transitions_added=982*4096*48,recorded_cumulative_transitions=final['transitions'],
        completed_update_transitions=4000*4096*48,
        inherited_abandoned_rollout_transitions=budget['inherited_abandoned_rollout_transitions'],
        optimizer_updates_added_each=19640,checkpoint_sha256=base.sha(LOG/'model_final.pt'),behavior_accepted=False))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','train'));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    if args.mode=='prepare':prepare(output);return
    budget=json.loads((output/'budget.json').read_text())
    assert base.sha(MODEL)==budget['checkpoint_sha256']
    for manifest in ('runtime-source-sha256.json','launcher-source-sha256.json'):
        for name,digest in json.loads((output/manifest).read_text()).items():assert base.sha(ROOT/name)==digest,name
    base.MODEL=MODEL;base.LOG=LOG
    if args.mode=='train':
        shutil.copy2(__file__,output/'resume-launcher.py')
        base.worker(output,'train');return
    try:
        assert not LOG.exists()
        base.run_command(output,'train',[str(base.PYTHON),'-u','-m',MODULE,'train','--output',str(output)],output/'process-result.json')
        audit(output)
        base.write(output/'status.json',dict(status='training_complete',source_iteration=3018,final_iteration=4000,
            iterations_added=982,checkpoint=str(LOG/'model_final.pt'),checkpoint_sha256=base.sha(LOG/'model_final.pt'),behavior_accepted=False))
    except BaseException as exc:
        base.write(output/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
