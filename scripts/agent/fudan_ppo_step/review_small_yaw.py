"""5200轮结束后执行GPU回归，保留4200基线，不自动换模。"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .launch_training import ROOT, PYTHON, sha, write
from .prepare_small_yaw import SOURCE_SHA

BASELINE=ROOT/'artifacts/fudan_ppo_step/20261118-v8-effect4200-01'


def compare(old, new):
    from scripts.agent.motion.acceptance import compare_baseline
    old_turn=json.loads((old/'turns/acceptance.json').read_text())
    new_turn=json.loads((new/'turns/acceptance.json').read_text())
    turns=compare_baseline(new_turn,old_turn)['comparison']
    new_rows=json.loads((new/'turns/metrics.json').read_text())
    old_rows={r['id']:r for r in json.loads((old/'turns/metrics.json').read_text())}
    focus=[dict(id=r['id'],command=r['yaw'],actual_yaw=r['actual_yaw_mean'],
        old_yaw_mae=old_rows[r['id']]['yaw_mae'],yaw_mae=r['yaw_mae'],passed=r['passed'])
        for r in new_rows if r['height']==.35 and r['vx']==0 and abs(r['yaw']) in (.5,1.)]
    assert sorted(r['command'] for r in focus)==[-1.,-.5,.5,1.]
    flat=json.loads((new/'flat47/acceptance/acceptance.json').read_text())
    assert flat['total']==47 and len(flat['comparison']['retained'])+len(flat['comparison']['regressed'])==47
    left,right=old/'stairs',new/'stairs'
    p,q=[json.loads((folder/'protocol.json').read_text()) for folder in (left,right)]
    for key in ('cases','seconds','command_height_m','high_speed','max_approach_speed',
                'action_delay_steps','sensor_delay_steps','friction'):
        assert p[key]==q[key],key
    initial_max=0.
    for case in p['cases']:
        traces=[folder/'traces'/(case['id']+'.npz') for folder in (left,right)]
        for path,protocol in zip(traces,(p,q)):
            assert sha(path)==protocol['trace_sha256'][case['id']]
        with np.load(traces[0]) as a,np.load(traces[1]) as b:
            initial_max=max(initial_max,float(abs(a['initial']-b['initial']).max()))
    assert initial_max<2e-6
    a,b=[{r['id']:r for r in json.loads((folder/'metrics.json').read_text()) if r['height']==.2}
         for folder in (left,right)]
    assert a.keys()==b.keys() and len(a)==27 and sum(r['task_score']['passed'] for r in a.values())==24
    lost=[key for key in a if a[key]['task_score']['passed'] and not b[key]['task_score']['passed']]
    strict_lost=[key for key in a if a[key]['parked'] and not b[key]['parked']]
    eligible=not lost and not turns['regressed'] and not flat['comparison']['regressed'] and all(
        r['passed'] and r['yaw_mae']<=.10 for r in focus)
    return dict(candidate_eligible_on_this_subset=eligible,full_task_accepted=False,
        focused_turns=focus,old_turn_regressions=turns['regressed'],initial_max=initial_max,
        stair_regressions=lost,strict_stair_regressions_diagnostic=strict_lost,
        stair_passed=sum(r['task_score']['passed'] for r in b.values()),stair_total=27,
        flat_passed=flat['technical_passed'],flat_regressions=flat['comparison']['regressed'],optimizer_updates=0)


def monitor(run,output,deadline):
    from .check import source_fingerprint
    output.mkdir(parents=True,exist_ok=False)
    state=output/'status.json'
    budget=json.loads((run/'budget.json').read_text())
    assert (budget['initial_iterations'],budget['iterations'],budget['num_envs'])==(4200,5200,4096)
    write(state,dict(status='waiting_for_5200'))
    try:
        while time.time()<deadline:
            path=run/'status.json'
            if path.exists():
                try:status=json.loads(path.read_text())
                except json.JSONDecodeError:status={}
                if status.get('status')=='completed':break
                if status.get('status')=='failed':raise RuntimeError(status)
            time.sleep(20)
        else:raise TimeoutError('Waiting for training audit timed out')
        assert source_fingerprint()==budget['source_fingerprint'], 'Evaluation source changed since training launch'
        audit=json.loads((run/'training-audit.json').read_text());checkpoint=Path(audit['checkpoint'])
        assert audit['passed'] and audit['iterations']==5200 and sha(checkpoint)==audit['checkpoint_sha256']
        for folder in ('turns','stairs'):
            assert json.loads((BASELINE/f'{folder}-4200/protocol.json').read_text())[
                'checkpoint_sha256' if folder=='turns' else 'source_sha256']==SOURCE_SHA
        old=output/'baseline';old.mkdir()
        for folder in ('turns','stairs'):
            (old/folder).symlink_to(BASELINE/f'{folder}-4200',target_is_directory=True)
        write(state,dict(status='evaluating',checkpoint=str(checkpoint),optimizer_updates=0))
        end=time.time()+1800
        def run_check(name,argv):
            with (output/f'{name}.log').open('x') as stream:
                subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    check=True,timeout=max(1,end-time.time()))
        for mode in ('turns','stairs'):
            run_check(mode,[str(PYTHON),'-m','scripts.agent.fudan_ppo_step.evaluate_fixed200_turn_effect',
                mode,'--checkpoint',str(checkpoint),'--output',str(output/mode),'--deadline',str(end)])
        run_check('flat47',[str(PYTHON),'-m','scripts.agent.fudan_ppo_step.inspect_training',
            'flat','--run',budget['log_dir'],'--checkpoint',str(checkpoint),'--output',str(output/'flat47')])
        run_check('accept',[str(ROOT/'scripts/mjlab.sh'),'accept','--task','fudan_ppo_walk',
            '--recording',str(output/'flat47/gpu'),'--output',str(output/'flat47/acceptance'),
            '--baseline',str(BASELINE/'flat47/acceptance/acceptance.json')])
        assert json.loads((output/'stairs/audit/audit.json').read_text())['passed']
        assert json.loads((output/'flat47/runtime-audit.json').read_text())['passed']
        assert sha(checkpoint)==audit['checkpoint_sha256'] and source_fingerprint()==budget['source_fingerprint']
        result=compare(old,output)
        write(output/'review.json',dict(checkpoint_sha256=audit['checkpoint_sha256'],**result))
        write(state,dict(status='evaluated',**result))
    except Exception as exc:
        write(state,dict(status='failed',error=repr(exc),restarted_training=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();monitor(args.run.resolve(),args.output.resolve(),args.deadline)
