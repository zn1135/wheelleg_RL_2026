"""Check selected inference against the previously executed probe, without physics or NN."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from ct_mjlab.fudan_ppo_walk import selected_policy as selected
from ct_mjlab.fudan_ppo_walk.protocol import samples
from . import evaluate


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    path=Path('artifacts/fudan_ppo_walk/fudan-pitch-reference-small-20261103/source/scripts/agent/fudan_ppo_walk/probe_pitch_reference.py')
    spec=importlib.util.spec_from_file_location('_executed_pitch_probe',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    checks=[]
    def verify(name,condition):
        assert condition,name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('No physics, network forward, CUDA initialization or training in selection checks')
    with patch('torch.nn.Linear.forward',side_effect=forbidden),patch('torch.optim.Adam.step',side_effect=forbidden),patch('torch.cuda._lazy_init',side_effect=forbidden),patch('mujoco.mj_step',side_effect=forbidden):
        record=selected.selection_record()
        verify('selected8000_hash_pitch2_no_integral',record['parking_pitch_deg']==2 and not record['integral_enabled'] and '7000_to8000' in record['checkpoint'])
        rewards=selected.FudanWalkCfg().rewards
        verify('split_roll_pitch_rewards_without_parking_velocity',len(rewards.scales)==17 and 'roll' in rewards.scales and 'pitch' in rewards.scales and 'orientation' not in rewards.scales and 'parking_velocity' not in rewards.scales)
        items=samples()[::21];torch.manual_seed(39)
        obs=torch.randn(len(items),25);history=torch.randn(len(items),125)
        before_obs=obs.clone();before_history=history.clone()
        starts=selected.reference_starts(items,obs);target=torch.full((len(items),),2*torch.pi/180.)
        for t in (0.,1.99,2.,2.01,2.25,2.5,6.99,7.,7.01,7.25,7.5,10.,15.):
            new=selected.policy_inputs(obs,history,t,starts,target)
            original=old.policy_inputs(obs,history,t,starts,target)
            verify('executed_probe_exact_'+str(t),all(torch.equal(a,b) for a,b in zip(new,original)))
        verify('original_inputs_not_mutated',torch.equal(obs,before_obs) and torch.equal(history,before_history))
        moving=[i for i,s in enumerate(items) if all(seg['command'][:2]!=[0.,0.] for seg in s['case']['segments'])]
        new=selected.policy_inputs(obs,history,10.,starts,target)
        verify('all_moving_or_turning_samples_exact',torch.equal(new[0][moving],obs[moving]) and torch.equal(new[1][moving],history[moving]))
        for i,s in enumerate(items):
            if s['case']['category'].startswith('stop_'):
                verify('stop_ramp_starts7_'+s['id'],starts[i].item()==7.)
        def cli(extra):
            argv=['evaluate','--output','unused','--deadline','9999999999',*extra]
            with patch.object(sys,'argv',argv),patch.object(evaluate,'gpu') as gpu:
                evaluate.main()
                return gpu.call_args.args[0]
        default=cli([])
        verify('default_cli_uses_selected_model_and_bias',default.checkpoint==Path(record['checkpoint']) and default.selection==record)
        explicit=cli(['--checkpoint',record['checkpoint']])
        verify('explicit_checkpoint_retains_raw_replay',explicit.selection is None and explicit.checkpoint==Path(record['checkpoint']))
        cfg=selected.FudanWalkCfg();cfg.inference.checkpoint_sha256='bad'
        with patch.object(selected,'FudanWalkCfg',return_value=cfg):
            rejected=False
            try:selected.selection_record()
            except ValueError:rejected=True
        verify('mismatched_selected_model_rejected',rejected)
        cfg=selected.FudanWalkCfg();cfg.parking_integral.enabled=True
        with patch.object(selected,'FudanWalkCfg',return_value=cfg):
            rejected=False
            try:selected.selection_record()
            except ValueError:rejected=True
        verify('integral_change_rejected',rejected)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(passed=True,checks=checks,count=len(checks),selection=record,
        physics_steps=0,model_forwards=0,optimizer_updates=0,
        scope='Executed probe arithmetic, fixed47-case schedule, default/explicit CLI routing; no new behavioral validation')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
