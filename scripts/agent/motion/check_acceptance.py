"""Check unified arithmetic, GPU guards, source migration and report behavior without dynamics."""
import argparse
import ast
import copy
from contextlib import ExitStack,redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from .acceptance import canonical_sample,canonical,envelope,compare_baseline,bc,smp,supermimic,isaac,STANDARD
from .assess import metrics_verdict,THRESHOLDS


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();checks=[]
    def verify(name,value):
        assert value,name
        checks.append(name)
    from ct_mjlab.validation import require_gpu
    from ct_mjlab.fudan_ppo_walk.protocol import samples
    from scripts.agent.fudan_ppo_walk.check_scoring import fixture
    from scripts.agent.fudan_ppo_walk.score_v2 import score as legacy
    import torch
    def forbidden(*a,**kw):raise AssertionError('Real physics, network forward or update forbidden')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as temporary:
        for name in ['mujoco.mj_step','mujoco.mj_forward','mjlab.envs.ManagerBasedRlEnv.step',
                     'torch.nn.Linear.forward','torch.optim.Adam.step']:
            stack.enter_context(patch(name,side_effect=forbidden))
        for device in ['cuda','cuda:0','cuda:1']:
            verify('gpu_allowed:'+device,require_gpu(device)['validation_backend']=='mujoco_warp_gpu')
        for device in ['cpu','mps']:
            try:require_gpu(device)
            except ValueError:rejected=True
            else:rejected=False
            verify('cpu_or_mps_rejected:'+device,rejected)
        from ct_mjlab.scripts.play import play
        from ct_mjlab.scripts.eval import evaluate_seed
        from ct_mjlab.supermimic.__main__ import play as sm_play
        from scripts.agent.imitation.eval_bc import run as bc_run
        from scripts.agent.supermimic.assess_chuanliantui_supermimic import run as sm_assess
        for name,call in [('play',lambda:play(SimpleNamespace(device='cpu'))),
                          ('smp_eval',lambda:evaluate_seed(SimpleNamespace(device='cpu'),42)),
                          ('supermimic_play',lambda:sm_play(SimpleNamespace(device='cpu'))),
                          ('bc_eval',lambda:bc_run(SimpleNamespace(device='cpu'))),
                          ('supermimic_assess',lambda:sm_assess(SimpleNamespace(device='cpu')))]:
            try:call()
            except ValueError as exc:rejected='GPU' in str(exc)
            else:rejected=False
            verify('real_entry_rejects_cpu_before_model_or_env:'+name,rejected)
        values=dict(vx_mae=.1,yaw_mae=.1,horizontal_peak=.05,pitch_peak=5.,complete=True,parking=True)
        verify('all_equal_boundaries_pass',metrics_verdict(**values)['passed'])
        for key in ['vx_mae','yaw_mae','horizontal_peak','pitch_peak']:
            bad=values.copy();bad[key]+=1e-5
            verify('above_boundary_fails:'+key,not metrics_verdict(**bad)['passed'])
        for key in ['vx_mae','yaw_mae','horizontal_peak','pitch_peak']:
            bad=values.copy();bad[key]=None
            verify('missing_metric_fails:'+key,not metrics_verdict(**bad)['passed'])
        items=samples();sample=next(s for s in items if s['case']['id']=='forward_2_0_0.26')
        clean=fixture(sample)
        verify('canonical_clean_pass',canonical_sample(sample,clean)['passed'])
        for key,value,column in [('linear_body',2.15,0),('angular_body',.15,2),('roll_pitch_deg',7.,1)]:
            bad=copy.deepcopy(clean);bad[key][1000:-1,column]=value
            verify('legacy_pass_new_fail:'+key,legacy(sample,bad)['passed'] and not canonical_sample(sample,bad)['passed'])
        stop=next(s for s in items if s['case']['id']=='stop_1_0.26');bad=fixture(stop)
        bad['linear_world'][4000:-1,0]=.075
        verify('parking_075_old_pass_new_fail',legacy(stop,bad)['passed'] and not canonical_sample(stop,bad)['passed'])
        bad=copy.deepcopy(clean);bad['numerical_failure'][3]=True
        verify('numerical_failure_retained',not canonical_sample(sample,bad)['passed'])
        verify('missing_trace_fields_fail',not canonical_sample(sample,{'time':clean['time']})['passed'])
        report=envelope('smp','fixture',[dict(passed=True)],{'parking'},True)
        verify('unmeasured_capabilities_not_passed',not report['technical_thresholds_met'] and set(report['missing_capabilities'])=={'forward','backward','turn'})
        report=envelope('bc','fixture',[],set(),True,rise=dict(passed=True))
        verify('rise_does_not_imply_motion_pass',not report['technical_thresholds_met'])
        report=envelope('supermimic','fixture',[dict(passed=True)],set(['forward','backward','turn','parking']),False)
        verify('unverified_backend_cannot_pass',not report['technical_thresholds_met'])
        report.update(task='smp',cohort_sha256='same',samples=[dict(sample_id='a',passed=True),dict(sample_id='b',passed=False)])
        new=copy.deepcopy(report);new['samples']=[dict(sample_id='a',passed=False),dict(sample_id='b',passed=True)]
        compared=compare_baseline(new,report)
        verify('new_success_cannot_cancel_regression',compared['comparison']['regressed']==['a'] and not compared['technical_thresholds_met'])
        for changed in ['cohort_sha256','standard']:
            bad=copy.deepcopy(report);bad[changed]='different'
            try:compare_baseline(bad,report)
            except ValueError:rejected=True
            else:rejected=False
            verify('different_baseline_rejected:'+changed,rejected)
        directory=Path(temporary);(directory/'traces').mkdir()
        protocol=dict(samples=[sample],mode='diagnostic_not_formal_acceptance',validation_backend='mujoco_warp_gpu')
        (directory/'protocol.json').write_text(json.dumps(protocol));np.savez_compressed(directory/'traces'/f"{sample['id']}.npz",**clean)
        result=canonical(directory,'fudan_ppo_walk')
        verify('diagnostic_not_full_acceptance',result['technical_passed']==1 and not result['technical_thresholds_met'])
        protocol['validation_backend']='native_cpu';(directory/'protocol.json').write_text(json.dumps(protocol))
        try:canonical(directory,'fudan_ppo_walk')
        except ValueError:rejected=True
        else:rejected=False
        verify('cpu_record_cannot_be_relabelled',rejected)
        protocol['validation_backend']='mujoco_warp_gpu';(directory/'protocol.json').write_text(json.dumps(protocol))
        (directory/'traces'/f"{sample['id']}.npz").unlink();result=canonical(directory,'fudan_ppo_walk')
        verify('missing_sample_keeps_denominator',result['total']==1 and result['technical_passed']==0)
        bc_dir=directory/'bc';bc_dir.mkdir();feature=np.zeros((2000,1,30));feature[:,:,3]=-1.;feature[:,:,4]=.02
        velocity=np.zeros((2000,1,24));velocity[:,:,0]=.02
        np.savez_compressed(bc_dir/'trace.npz',time=np.arange(1,2001)*.01,features=feature,qvel=velocity,
            command=np.zeros((2000,1,3)),alive=np.ones((2000,1),dtype=bool))
        bc_meta=dict(validation_backend='mujoco_warp_gpu',trace_sha256=hashlib.sha256((bc_dir/'trace.npz').read_bytes()).hexdigest(),
            rows=[dict(env=0,full_20s=True,finite=True,final_balance_v31=True,strict_rise=False)],strict_rise=0)
        (bc_dir/'result.json').write_text(json.dumps(bc_meta));result=bc(bc_dir)
        verify('bc_parking_uses_common_metrics',result['technical_passed']==1 and not result['technical_thresholds_met'])
        verify('bc_rise_kept_separate',result['rise']['legacy_strict_rise']==0)
        smp_dir=directory/'smp';smp_dir.mkdir();(smp_dir/'result.json').write_text(json.dumps(dict(validation_backend='mujoco_warp_gpu',strict_recovered=100)))
        verify('smp_recovery_not_motion_acceptance',not smp(smp_dir)['technical_thresholds_met'])
        legacy_path=directory/'isaac.json';legacy_path.write_text(json.dumps(dict(metadata=dict(validation_backend='physx_gpu',task='chuanliantui_standup'),summary={})))
        verify('legacy_gpu_report_keeps_missing_motion',isaac(legacy_path)['validation_backend']=='physx_gpu' and not isaac(legacy_path)['technical_thresholds_met'])
        sm_dir=directory/'supermimic';sm_dir.mkdir()
        (sm_dir/'summary.json').write_text(json.dumps(dict(validation_backend='mujoco_warp_gpu',summary=dict(per_environment=[
            dict(env=0,speed_m_s=.5,yaw_rate_rad_s=0.,survived=True,final_balanced=False)]))))
        with patch('scripts.agent.motion.acceptance.reevaluate',return_value=dict(rows=[dict(env=0,passed=True,phases={'cruise':{},'parking':{}})])):
            result=supermimic(sm_dir)
        verify('supermimic_motion_cannot_override_balance_failure',result['technical_passed']==0)
        # Run only the added legacy guard AST; do not import Isaac Gym into mjlab Python.
        tree=ast.parse(Path('sim2sim/eval_isaac.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        guard=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and 'Final validation requires GPU' in ast.unparse(n))
        block=ast.Module(body=fn.body[guard:guard+3],type_ignores=[])
        good=SimpleNamespace(sim_device='cuda:0',rl_device='cuda:0',use_gpu=False,use_gpu_pipeline=False)
        exec(compile(block,'legacy-guard','exec'),{'args':good})
        verify('isaac_guard_uses_gpu_pipeline',good.use_gpu and good.use_gpu_pipeline)
        for sim,rl in [('cpu','cuda:0'),('cuda:0','cpu')]:
            try:exec(compile(block,'legacy-guard','exec'),{'args':SimpleNamespace(sim_device=sim,rl_device=rl)})
            except ValueError:rejected=True
            else:rejected=False
            verify('isaac_cpu_rejected:'+sim+':'+rl,rejected)
    before=args.baseline/'before';name='mjlab_training/src/ct_mjlab/supermimic/__main__.py'
    def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    verify('supermimic_train_function_unchanged',ast.dump(function(before/name,'train'),include_attributes=False)==ast.dump(function(Path(name),'train'),include_attributes=False))
    protected=json.loads((args.baseline/'protected.json').read_text())
    for path,digest in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    verify('training_physics_v2_scoring_protected',True)
    result=dict(passed=True,count=len(checks),checks=checks,standard=STANDARD,thresholds=THRESHOLDS,
        physics_steps=0,model_forwards=0,optimizer_updates=0,legacy_execution='AST guard only; no Isaac imports or dynamics')
    with args.output.open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__':main()
