"""Run the existing frozen walking evaluator and bind its recording to the candidate PT."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.agent.fudan_ppo_walk.evaluate import gpu


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();a.selection=None;a.response_parameters=None;a.leg_load_profile='legacy';a.diagnostic_height=None
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();torch.set_num_threads(1)
    with patch('torch.optim.Adam.step',side_effect=AssertionError('Regression cannot train')):
        gpu(a)
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=digest:
        raise ValueError('Checkpoint changed during walking regression')
    (a.output/'step-policy-receipt.json').write_text(json.dumps(dict(
        checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=digest,optimizer_updates=0,
        evaluator='scripts.agent.fudan_ppo_walk.evaluate',diagnostic=a.diagnostic),indent=2)+'\n')


if __name__=='__main__':main()
