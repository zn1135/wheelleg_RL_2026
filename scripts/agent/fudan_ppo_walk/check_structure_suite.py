"""Run the authorized nonphysical structural suite under process-wide no-physics/no-update guards."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GUARD='''import sys,runpy
from contextlib import ExitStack
from unittest.mock import patch
module=sys.argv[1];sys.argv=sys.argv[1:]
def forbidden(*a,**kw):raise AssertionError('Structure suite forbids physical construction/integration and optimizer updates')
with ExitStack() as guards:
 for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','mjlab.envs.ManagerBasedRlEnv.__init__','mjlab.envs.ManagerBasedRlEnv.step','torch.optim.Adam.step'):
  guards.enter_context(patch(name,side_effect=forbidden))
 runpy.run_module(module,run_name='__main__')
'''


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source='artifacts/fudan-structure-alignment-20261103/before/mjlab_training/src/ct_mjlab/fudan_ppo_walk'
    parent='logs/mjlab/fudan_ppo_walk/20261102_resume6000_to7000/model_final.pt'
    cases=[('structure','check_structure_alignment',[]),('algorithm','check_algorithm_alignment',[]),('tools','check_structure_tools',[]),
        ('history-v3','check_history_v3',[]),('reset-v2','check_reset_v2',[]),('parking','check_parking_commands',[]),
        ('disturbances','check_disturbances',[]),('integral','check_parking_integral',[]),
        ('preserve-v2','check_preserve_v2',['--checkpoint',parent]),
        ('migration-v2-v3','check_history_migration',['--checkpoint',parent,'--baseline-source',source]),
        ('resume','check_resume',['--checkpoint','logs/mjlab/fudan_ppo_walk/20261102_reset_v2_iter3000/model_final.pt']),
        ('budget','check_iteration_budget',[]),('logging','check_reward_logging',['--baseline','artifacts/fudan-reward-logging-20261102/before-runner.py']),
        ('gpu-entry','check_gpu_validation',['--migration','artifacts/fudan-gpu-only-validation-20261102']),
        ('scoring','check_scoring_v2',[]),('reference','check',[])]
    def run(case):
        name,module,extra=case;directory=a.output/name;directory.mkdir();result=directory/'result.json'
        destination=directory/'static' if module=='check' else result
        cmd=[sys.executable,'-c',GUARD,'scripts.agent.fudan_ppo_walk.'+module,*extra,'--output',str(destination)]
        with (directory/'output.log').open('w') as log:process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        receipt=dict(name=name,exit_code=process.returncode,result=str(destination/'report.json' if module=='check' else result))
        print(json.dumps(receipt),flush=True);return receipt
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,cases))
    record=dict(passed=all(r['exit_code']==0 for r in results),checks=results,physics_steps=0,optimizer_updates=0,
        physical_environment_construction='forbidden_by_guard',training='not_started',gpu_evaluation='not_started')
    (a.output/'suite.json').write_text(json.dumps(record,indent=2)+'\n')
    if not record['passed']:raise SystemExit(1)

if __name__=='__main__':main()
