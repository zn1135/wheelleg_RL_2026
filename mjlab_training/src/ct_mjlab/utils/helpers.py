"""Argument definitions for the stable ct-mjlab command line."""
import argparse
from pathlib import Path

from ..envs.chuanliantui.profiles import TRAINING_PROFILES
from ..envs import task_registry


def build_parser():
    parser=argparse.ArgumentParser(description='Entrypoints for the independent mjlab chuanliantui task.')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('train','play','eval','export','check'):
        p=sub.add_parser(name)
        p.add_argument('--task',choices=task_registry.names(),default='chuanliantui')
        p.add_argument('--seed',type=int,default=42)
        if name in ('train','play','check'):
            p.add_argument('--num-envs',type=int,default=4096 if name=='train' else 1)
            p.add_argument('--device',default='cuda:0')
        if name in ('train','check'):
            p.add_argument('--training-profile', choices=TRAINING_PROFILES,
                default='huananhu-v14', help='Original SCUT V14 Flat-v0 (default), r15 curriculum or r14 recover')
        if name in ('play','eval','export'):p.add_argument('--checkpoint',type=Path,required=name=='export')
        if name in ('play','eval'):
            p.add_argument('--seconds',type=float,default=30)
            p.add_argument('--render',action='store_true');p.add_argument('--report',type=Path)
        if name=='play':
            p.add_argument('--backend',choices=['mjlab','cpu'],default='mjlab')
            p.add_argument('--random-reset',action='store_true',help='Preview the training pose distribution')
        if name=='export':p.add_argument('--output',type=Path,required=True)
        if name=='train':
            p.add_argument('--iterations',type=int,default=20000);p.add_argument('--run-name',default='')
            p.add_argument('--resume',type=Path)
            p.add_argument('--resume-v14-wheel-support', action='store_true',
                help='Explicit known recovery model2999 migration to wheel-to-ground load-bearing goals')
            p.add_argument('--resume-v14-recovery-fix', action='store_true',
                help='Explicit evaluated 2000-update height model migration to recovery and height curriculum')
            p.add_argument('--resume-v14-stand-fix', action='store_true',
                help='Explicit evaluated V14 height model_1999 migration to the standing phase')
            p.add_argument('--resume-v14-height-fix', action='store_true',
                help='Explicit evaluated V14 model_999 migration: height range and velocity height gate')
            p.add_argument('--resume-contact-force-fix', action='store_true',
                help='Historical v1 to v2 migration; use the archived v2 release')
            p.add_argument('--resume-pose-bias-fix', action='store_true',
                help='Explicit known assisted-r15 v2 to v3 removal of pose_bias in all stages')
            p.add_argument('--resume-orientation-weight', action='store_true',
                help='Explicit r15-v3 migration with stronger orientation weights and unchanged formula')
            p.add_argument('--resume-wide-height-gate',action='store_true',
                help='Archived r11 to r13 gate-only migration; unavailable in current r14')
            p.add_argument('--resume-disable-knee-limit',action='store_true',
                help='Historical r11 to r12 knee-only migration; unavailable in current r14')
            p.add_argument('--render',action='store_true',help='Show the actual PPO environments in a native grid')
            p.add_argument('--viewer-fps',type=float,default=30)
        if name=='check':p.add_argument('--steps',type=int,default=32)
    return parser
