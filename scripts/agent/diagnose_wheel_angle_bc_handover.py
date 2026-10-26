#!/usr/bin/env python3
"""Zero-update diagnostic: teacher starts the robot, then the BC actor takes over."""
import argparse,json,shutil
from dataclasses import dataclass,field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import eval_bc
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction
from check_lqr_student_closed_loop import BridgeCfg,ClosedLoopAction,digest


def diagnose(args):
    torch.set_num_threads(1)
    selected=[];applied=[]
    class HandoverAction(ClosedLoopAction):
        def process_actions(self,actions):
            super().process_actions(actions)
            active=self._substep_ticks*self._env.physics_dt<args.teacher_prefix_s-1e-7
            targets=torch.where(active[:,None],self.raw_action,actions)
            HuananhuPdAction.process_actions(self,targets)
            selected.append(active.cpu().numpy().copy());applied.append(self.raw_action.cpu().numpy().copy())
    @dataclass(kw_only=True)
    class HandoverCfg(BridgeCfg):
        class_type:type=field(default_factory=lambda:HandoverAction)
    get_cfgs=eval_bc.task_registry.get_cfgs
    def diagnostic_cfg(*a,**k):
        cfg,training=get_cfgs(*a,**k)
        cfg.actions['pd']=HandoverCfg(**{name:value for name,value in vars(cfg.actions['pd']).items() if name!='class_type'})
        return cfg,training
    call=SimpleNamespace(dataset=args.dataset,corrections=args.corrections,checkpoint=args.checkpoint,
        pose_dataset=args.pose_dataset,device='cuda:0',split='validation',collect_corrections=False,
        full_precision=False,output=args.output)
    with patch.object(eval_bc.task_registry,'get_cfgs',side_effect=diagnostic_cfg):eval_bc.run(call)
    report=json.loads((args.output/'result.json').read_text())
    with np.load(args.output/'trace.npz') as d:arrays={k:d[k] for k in d.files}
    arrays.update(teacher_selected=np.asarray(selected),applied_action=np.asarray(applied))
    alive=arrays['alive'];early=(arrays['time']<=args.teacher_prefix_s)[:,None]&alive
    assert arrays['teacher_selected'][early].all()
    late=(arrays['time']>args.teacher_prefix_s)[:,None]&alive
    assert not arrays['teacher_selected'][late].any()
    np.testing.assert_array_equal(arrays['action'][late],arrays['applied_action'][late])
    np.savez_compressed(args.output/'trace.npz',**arrays)
    report.update(diagnostic_only=True,independent_student_control=False,
        teacher_prefix_s=args.teacher_prefix_s,trace_sha256=digest(args.output/'trace.npz'),
        diagnostic_script_sha256=digest(Path(__file__)),
        scope='Hybrid teacher prefix followed by actual actor control. Diagnostic only, not independent rise, not a training demonstration, no optimizer or heldout labels.')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    target=args.output/'snapshot'/Path(__file__).name;target.parent.mkdir();shutil.copy2(__file__,target)
    print(json.dumps(dict(diagnostic_only=True,optimizer_updates=0,teacher_prefix_s=args.teacher_prefix_s,
        hybrid_final_balance=report['final_balance'],hybrid_parking=report['qualified'],num_envs=report['num_envs']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','pose-dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',default=[])
    p.add_argument('--teacher-prefix-s',type=float,default=2.);diagnose(p.parse_args())
