"""Prepare provenance, then enforce one 600s experiment with a 360s replay cutoff."""
import argparse,json,time,subprocess,signal,os,shutil,hashlib
from pathlib import Path
from datetime import datetime
from ct_mjlab.envs.fudan_ppo_walk.parking_integral import configuration
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg

CHECKPOINT=Path('logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt')
RECORDING=Path('artifacts/fudan-gpu-replay-8000-20261102')
EXPECTED_SHA='39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output;root.mkdir(parents=True,exist_ok=True)
    assert not(root/'budget.json').exists(),'Never reuse or extend a started budget'
    check=json.loads((root/'checks.json').read_text());assert check['passed'] and check['physics_steps']==check['model_forwards']==0
    assert all(sha(p)==value for p,value in check['source_sha256'].items())
    assert sha(CHECKPOINT)==EXPECTED_SHA
    for height in ('h022','h026','h030'):
        source=RECORDING/height/'gpu15';protocol=json.loads((source/'protocol.json').read_text());assert len(protocol['samples'])==15
        for s in protocol['samples']:
            assert(source/'fixtures'/(s['id']+'_initial.npz')).is_file() and (source/'traces'/(s['id']+'.npz')).is_file()
    files=[*Path('mjlab_training/src/ct_mjlab/fudan_ppo_walk').glob('*.py'),
        *Path('mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk').glob('*.py'),
        *Path('mjlab_training/src/ct_mjlab/rsl_rl').rglob('*.py'),*Path('scripts/agent/fudan_ppo_walk').glob('*.py'),
        *Path('scripts/agent/motion').glob('*.py'),Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'),
        *Path('mjlab_training/src/ct_mjlab/envs/chuanliantui').glob('*.py')]
    manifest={str(path):sha(path) for path in files}
    for path in files:
        dest=root/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    (root/'source-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    inputs={str(path):sha(path) for path in RECORDING.glob('h*/gpu15/**/*') if path.is_file()}
    (root/'input-sha256.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (root/'config.json').write_text(json.dumps(dict(task_config={name:{k:v for k,v in vars(value).items() if not k.startswith('_')} for name,value in vars(FudanWalkCfg).items() if not name.startswith('_')},default=configuration(),branches=['disabled','enabled_truth_feedback','enabled_encoder_feedback'],disturbances='all_off',raw_command_scoring=True,model_sha256=EXPECTED_SHA),indent=2)+'\n')
    (root/'git-status.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    env=os.environ.copy();env['PYTHONPATH']='mjlab_training/src';env['MUJOCO_GL']='egl'
    python='mjlab_training/.venv/bin/python';steps=[]
    # No checkpoint load, policy construction, CUDA initialization or simulation before this boundary.
    started=time.time();mono=time.monotonic();deadline=started+600;replay_stop=started+360
    budget=dict(start_unix=started,replay_stop_unix=replay_stop,deadline_unix=deadline,maximum_wall_s=600,replay_maximum_wall_s=360,
        start_local=datetime.fromtimestamp(started).astimezone().isoformat(),deadline_local=datetime.fromtimestamp(deadline).astimezone().isoformat(),
        checkpoint=str(CHECKPOINT),checkpoint_sha256=EXPECTED_SHA,authorization='用户明确授权8000轮停车积分三组135轨迹，600秒唯一预算；不训练、不追加、不重试')
    (root/'budget.json').write_text(json.dumps(budget,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(budget,ensure_ascii=False),flush=True)
    def stage(name,args,limit):
        remaining=limit-(time.monotonic()-mono)
        if remaining<=0:
            result=dict(name=name,skipped='deadline',finished_elapsed_s=time.monotonic()-mono);steps.append(result)
            return result
        t=time.monotonic()
        with (root/(name+'.log')).open('w') as log:
            child=subprocess.Popen([python,'-m',*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        result=dict(name=name,pid=child.pid,exit_code=code,hard_timeout=timed_out,wall_s=time.monotonic()-t,finished_elapsed_s=time.monotonic()-mono)
        steps.append(result);(root/'pipeline-progress.json').write_text(json.dumps(steps,indent=2)+'\n');print(json.dumps(result),flush=True)
        return result
    replay=stage('replay',['scripts.agent.fudan_ppo_walk.diagnose_parking_integral','--checkpoint',str(CHECKPOINT),'--recording',str(RECORDING),'--output',str(root)],360)
    # Partial traces are still scored; no replay retry is permitted.
    report=stage('report',['scripts.agent.fudan_ppo_walk.report_parking_integral','--experiment',str(root),'--recording',str(RECORDING)],600)
    has_traces=bool(list(root.glob('h*/*/traces/*.npz')))
    if has_traces:video=stage('video',['scripts.agent.fudan_ppo_walk.render_parking_integral','--experiment',str(root)],600)
    else:
        video=dict(name='video',skipped='no_replay_trajectories',finished_elapsed_s=time.monotonic()-mono);steps.append(video)
        (root/'pipeline-progress.json').write_text(json.dumps(steps,indent=2)+'\n')
    reports=[json.loads(p.read_text()) for p in root.glob('h*/*/acceptance/acceptance.json')]
    result=dict(start_unix=started,finished_unix=time.time(),wall_s=time.monotonic()-mono,steps=steps,
        checkpoint_unchanged=sha(CHECKPOINT)==EXPECTED_SHA,source_unchanged=all(sha(p)==value for p,value in manifest.items()),
        input_unchanged=all(sha(p)==value for p,value in inputs.items()),optimizer_updates=0,
        final_acceptance='pending_user_video_review' if has_traces else 'not_ready_no_dynamic_results',no_retry=True,
        expected_trajectories=135,trajectory_files=len(list(root.glob('h*/*/traces/*.npz'))),complete_traces=sum(sum(bool(s.get('complete')) for s in r['samples']) for r in reports),
        compute_finished_elapsed_s=max((s['finished_elapsed_s'] for s in steps if 'skipped' not in s),default=0.),
        within_budget=all(s['finished_elapsed_s']<= (360 if s['name']=='replay' else 600)+.5 for s in steps if 'skipped' not in s))
    (root/'pipeline-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
