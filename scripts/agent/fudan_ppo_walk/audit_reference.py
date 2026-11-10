"""Audit the vendored algorithm text, allowing only package-local import changes."""
import ast,json,hashlib
from pathlib import Path
from ct_mjlab.fudan_ppo_walk.contract import WEIGHTS,PPO_CONFIG


def main():
    root=Path(__file__).resolve().parents[3]
    reference=root/'artifacts/fudan_ppo_walk/fudan-ppo-walk-20261101/reference/plane/wheel_legged_gym'
    current=root/'mjlab_training/src/ct_mjlab/fudan_ppo_walk'
    compared=[]
    for src,dst in [('modules/actor_critic_sequence.py','network.py'),('algorithms/ppo.py','ppo.py'),
                    ('storage/rollout_storage.py','storage.py'),('utils/utils.py','utils.py')]:
        text=(reference/'rsl_rl'/src).read_text()
        text=text.replace('from wheel_legged_gym.rsl_rl.modules import ActorCritic','from .network import ActorCriticSequence as ActorCritic').replace('from wheel_legged_gym.rsl_rl.storage import RolloutStorage','from .storage import RolloutStorage').replace('from wheel_legged_gym.rsl_rl.utils import split_and_pad_trajectories','from .utils import split_and_pad_trajectories')
        assert (current/dst).read_text()==text,dst
        compared.append(dst)
    cfg=ast.parse((reference/'envs/base/legged_robot_config.py').read_text())
    env=next(n for n in cfg.body if isinstance(n,ast.ClassDef) and n.name=='LeggedRobotCfg')
    rewards=next(n for n in env.body if isinstance(n,ast.ClassDef) and n.name=='rewards')
    scales=next(n for n in rewards.body if isinstance(n,ast.ClassDef) and n.name=='scales')
    values={n.targets[0].id:ast.literal_eval(n.value) for n in scales.body if isinstance(n,ast.Assign)}
    assert values==WEIGHTS
    ppo=next(n for n in cfg.body if isinstance(n,ast.ClassDef) and n.name=='LeggedRobotCfgPPO')
    algorithm=next(n for n in ppo.body if isinstance(n,ast.ClassDef) and n.name=='algorithm')
    params={n.targets[0].id:ast.literal_eval(n.value) for n in algorithm.body if isinstance(n,ast.Assign)}
    assert params==PPO_CONFIG
    print(json.dumps(dict(algorithm_modules_byte_equal_after_import_translation=compared,
        reference_weights_equal=True,reference_ppo_hyperparameters_equal=True,
        physics_steps=0,optimizer_updates=0,scope='Algorithm/weights only; environment adaptation listed separately'),indent=2))

if __name__=='__main__':main()
