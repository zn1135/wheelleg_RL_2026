"""Check GPU-only validation routing/geometry/export dependencies; no physics or NN forwards."""
import argparse
import ast
from contextlib import ExitStack,redirect_stdout,redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.fudan_ppo_walk import policy,export
from ct_mjlab.fudan_ppo_walk.protocol import samples,manifest
from scripts.agent.maintenance.check_sim2sim_artifact_paths import matches_baseline
from . import evaluate,precheck


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--migration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path.cwd();checks=[]
    def verify(name,value):
        assert value,name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('Physics/model forward/optimizer forbidden')
    with ExitStack() as stack:
        for name in ['mujoco.mj_step','mujoco.mj_forward','mjlab.envs.ManagerBasedRlEnv.step',
                     'torch.nn.Linear.forward','torch.optim.Adam.step']:
            stack.enter_context(patch(name,side_effect=forbidden))
        for engine in [[],['gpu']]:
            argv=['evaluate',*engine,'--checkpoint','model.pt','--output','unused','--deadline','123']
            with patch.object(sys,'argv',argv),patch.object(evaluate,'gpu') as gpu:
                evaluate.main()
            verify('default_gpu' if not engine else 'explicit_gpu',gpu.call_count==1 and gpu.call_args.args[0].engine=='gpu')
        for extra in [['cpu'],['gpu','--fixtures','old'],['gpu','--workers','4'],['gpu','--integration-substeps','4']]:
            argv=['evaluate',*extra,'--checkpoint','model.pt','--output','unused','--deadline','123']
            with patch.object(sys,'argv',argv),patch.object(evaluate,'gpu',side_effect=forbidden),redirect_stderr(io.StringIO()):
                try:evaluate.main()
                except SystemExit as exc:rejected=exc.code==2
                else:rejected=False
            verify('removed_cli_rejected:'+str(extra),rejected)
        model=SimpleNamespace(geom_bodyid=np.array([1,1]),geom_contype=np.array([1,0]),
            geom_conaffinity=np.array([0,0]),geom_type=np.array([7,7]),geom_dataid=np.array([0,0]),
            mesh_vertadr=np.array([0]),mesh_vertnum=np.array([2]),mesh_vert=np.array([[0.,0.,0.],[0.,0.,.1]]))
        rotation=torch.eye(3).repeat(2,2,1,1);rotation[1,0]=torch.diag(torch.tensor([-1.,1.,-1.]))
        position=torch.tensor([[[0.,0.,.02],[0.,0.,-2.]],[[6.,3.,.12],[6.,3.,-2.]]])
        env=SimpleNamespace(num_envs=2,robot=SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([1]))),
            env=SimpleNamespace(sim=SimpleNamespace(mj_model=model,data=SimpleNamespace(
                geom_xpos=position,geom_xmat=rotation,qpos=torch.tensor([[0.,0.,.3],[6.,3.,.3]])))))
        rows=precheck.initial_geometry(env,[{'id':'a'},{'id':'b'}])
        verify('gpu_geometry_rotates_and_excludes_visual_mesh',all(abs(r['minimum_collision_mesh_height_m']-.02)<1e-7 for r in rows))
        verify('gpu_geometry_tracks_sample_ids',[r['id'] for r in rows]==['a','b'])
        before=args.migration/'before/mjlab_training/src/ct_mjlab/fudan_ppo_walk/native.py'
        def function(path,name):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        from ct_mjlab.fudan_ppo_walk.source_migration import legacy_source_directory
        sealed=legacy_source_directory()
        verify('historical_policy_loader_ast_and_known_migration',ast.dump(function(before,'load_policy'),include_attributes=False)==
            ast.dump(function(sealed/'policy.py','load_policy'),include_attributes=False))
        verify('export_uses_shared_policy_loader',export.load_policy is policy.load_policy)
        verify('full_matrix_unchanged',len(samples())==987 and len({s['case']['id'] for s in samples()})==47)
        verify('diagnostic_height030_still15',len(evaluate.evaluation_items(True,.30))==15)
        verify('evaluator_has_no_native_functions',not hasattr(evaluate,'cpu') and not hasattr(evaluate,'cpu_sample'))
        retired=json.loads((args.migration/'retired.json').read_text())
        for name in retired:
            verify('retired_absent:'+name,not Path(name).exists())
            verify('retired_source_archived:'+name,(args.migration/'before'/name).is_file())
        for folder in ['mjlab_training/src/ct_mjlab/fudan_ppo_walk','scripts/agent/fudan_ppo_walk']:
            for path in Path(folder).glob('*.py'):
                tree=ast.parse(path.read_text());compile(tree,str(path),'exec')
                for node in ast.walk(tree):
                    if isinstance(node,ast.ImportFrom):assert node.module not in ['ct_mjlab.fudan_ppo_walk.native','native'],path
        verify('all_active_fudan_modules_compile_without_native_imports',True)
        from ct_mjlab.fudan_ppo_walk.source_migration import verify_migration, historical_path_matches
        mapped=verify_migration()['files']
        protected=json.loads((args.migration/'protected.json').read_text())
        changes=json.loads(Path('docs/maintenance/20261101-sim2sim-protection.json').read_text()).get('authorized_updates',[])
        for name,digest in protected.items():
            actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            assert matches_baseline(Path.cwd()/name,digest) or any(c['path']==name and c['before']==digest and c['after']==actual for c in changes) or historical_path_matches(name,digest),name
        verify('protected_sources_unchanged_or_explicit_gpu_entry_update',True)
    result=dict(passed=True,checks=checks,count=len(checks),matrix_sha256=manifest()['sha256'],
        physics_steps=0,model_forwards=0,optimizer_updates=0,
        scope='GPU CLI, synthetic mesh geometry, unchanged loader/export wiring and source/archive checks; no behavior validation')
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
