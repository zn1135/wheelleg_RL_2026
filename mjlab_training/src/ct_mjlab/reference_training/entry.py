"""Local entry for the imported manager-based recover training task."""
from dataclasses import asdict, is_dataclass
import copy
import hashlib
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from .huananhu_env_cfg import make_huananhu_env_cfg, make_huananhu_ppo_cfg

INTERFACE = 'chuanliantui-mjlab-wheelleg-recover-r14'

def serializable(value):
    if is_dataclass(value): return {k:serializable(v) for k,v in asdict(value).items()}
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    if callable(value):return f'{value.__module__}.{value.__qualname__}'
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return str(value)

def runner_cfg(seed=42):
    cfg=make_huananhu_ppo_cfg();cfg.seed=seed
    result=asdict(cfg)
    for key in ('actor','critic'):
        for optional in ('cnn_cfg','rnn_type','rnn_hidden_dim','rnn_num_layers'):
            result[key].pop(optional,None)
        if result[key]['distribution_cfg'] is None:result[key].pop('distribution_cfg')
    return result

def manifest():
    from ..config import ASSET
    from .model import ACTUATED_JOINTS, POLICY_JOINTS
    root=Path(__file__).parent
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    cfg=make_huananhu_env_cfg()
    from .control_modes import PARAMETERS
    from .huananhu_rewards import REWARD_WEIGHTS, HEIGHT_GATE
    return dict(interface=INTERFACE, actor_dim=35,critic_dim=78,action_dim=6,
        training_rewards=dict(profile='wheelleg-recover-v14-wide-height-gate-velocity-heading',
            knee_limit_weight=REWARD_WEIGHTS['knee_limit'],height_gate=list(HEIGHT_GATE),
            weights=REWARD_WEIGHTS.copy()),
        modes=dict(semantics='dynamic-normal-recover',parameters=PARAMETERS,
            layout=['normal','stair','slope','recover','jump','height_target','state_time'],
            height_target='recover: current height command * 5; normal: 0',
            state_time='recover elapsed seconds; normal: 0'),
        joints=POLICY_JOINTS,observation_motor_order=list(POLICY_JOINTS),
        physical_motor_order=ACTUATED_JOINTS,
        policy_to_physical=[POLICY_JOINTS.index(name) for name in ACTUATED_JOINTS],
        asset_sha256=digest(ASSET),poses_sha256=digest(root.parent/'poses.json'),
        reset_bank_sha256=digest(root.parent/'reset_poses.npz'),
        source=json.loads((root/'SOURCE_IMPORT.json').read_text()),
        kernels_sha256={p.name:digest(p) for p in sorted(root.glob('*.py'))},
        environment=serializable(cfg),ppo=runner_cfg())

def validate_wide_height_gate_resume(saved):
    """Explicit r11 -> r13 transition: widen the gate and disable knee reward."""
    from ..policy import check_manifest
    from .huananhu_observations import OBSERVATION_CONTRACT
    from .pd_action import CONTROL_CONTRACT
    infos = saved.get('infos') or {}
    previous = infos.get('interface') or {}
    if previous.get('interface') != 'chuanliantui-mjlab-wheelleg-recover-r11':
        raise ValueError('Wide-gate migration requires a complete historical local r11 checkpoint')
    check_manifest(previous)
    if (infos.get('control_contract') != CONTROL_CONTRACT or
        infos.get('observation_contract') != OBSERVATION_CONTRACT or
        infos.get('experiment_profile_contract') !=
        'chuanliantui-huananhu-flat-35-78-ppo-sourcexml-spring90mm-velocity-reciprocal-knee-limit-mixed-reset-r11'):
        raise ValueError('r11 control, observation or experiment contract mismatch')
    current = manifest()
    if current['interface'] != 'chuanliantui-mjlab-wheelleg-recover-r13':
        raise ValueError('Wide-gate-only migration belongs to archived r13; r14 also changes velocity commands')
    if (current['training_rewards']['height_gate'] != [.05, .10] or
        current['training_rewards']['knee_limit_weight'] != 0.):
        raise ValueError('Wide-gate migration requires 5–10 cm and knee_limit=0')
    expected = copy.deepcopy(previous)
    expected['interface'] = current['interface']
    expected['training_rewards']['profile'] = current['training_rewards']['profile']
    expected['training_rewards']['height_gate'] = [.05, .10]
    expected['training_rewards']['knee_limit_weight'] = 0.
    expected['training_rewards']['weights']['knee_limit'] = 0.
    expected['environment']['rewards']['knee_limit']['weight'] = 0.
    for term in expected['environment']['rewards'].values():
        term['params']['height_gate'] = [.05, .10]
    # Only the gate plumbing, reward and explicit migration kernels changed.
    for name in ('entry.py', 'huananhu_rewards.py', 'huananhu_runner.py',
                 'huananhu_env_cfg.py', 'evaluate.py'):
        expected['kernels_sha256'][name] = current['kernels_sha256'][name]
    if json.loads(json.dumps(expected)) != json.loads(json.dumps(current)):
        raise ValueError('Migration permits only 3–6 -> 5–10 cm and knee_limit -0.5 -> 0')
    for name in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter'):
        if name not in saved:
            raise ValueError(f'Full resume is missing {name}')
    iteration = saved['iter']
    effective = infos.get('huananhu_training_iteration')
    if (not isinstance(iteration, int) or not isinstance(effective, int) or
        iteration < 0 or effective < iteration + 1):
        raise ValueError('Invalid completed-update counters in r11 checkpoint')
    return infos


def validate_knee_limit_resume(saved):
    """Allow only the explicitly requested r11 -> r12 knee-reward transition."""
    if manifest()['training_rewards']['height_gate'] != [.03, .06]:
        raise ValueError('r11 -> r12 knee-only migration cannot change the r13 height gate; use the historical r12 source')
    from ..policy import check_manifest
    from .huananhu_observations import OBSERVATION_CONTRACT
    from .pd_action import CONTROL_CONTRACT
    infos=saved.get('infos') or {}
    previous=infos.get('interface') or {}
    if previous.get('interface')!='chuanliantui-mjlab-wheelleg-recover-r11':
        raise ValueError('Knee-limit migration requires a historical local r11 checkpoint')
    check_manifest(previous)
    if (infos.get('control_contract')!=CONTROL_CONTRACT or
        infos.get('observation_contract')!=OBSERVATION_CONTRACT or
        infos.get('experiment_profile_contract')!=
        'chuanliantui-huananhu-flat-35-78-ppo-sourcexml-spring90mm-velocity-reciprocal-knee-limit-mixed-reset-r11'):
        raise ValueError('r11 control, observation or experiment contract mismatch')
    current=manifest();expected=copy.deepcopy(previous)
    if previous['environment']['rewards']['knee_limit']['weight']!=-.5:
        raise ValueError('Historical r11 knee-limit weight must be -0.5')
    if current['environment']['rewards']['knee_limit']['weight']!=0.:
        raise ValueError('Knee-limit migration requires the current reward weight to be zero')
    expected['interface']=current['interface']
    expected['training_rewards']['profile']=current['training_rewards']['profile']
    expected['training_rewards']['knee_limit_weight']=0.
    expected['training_rewards']['weights']['knee_limit']=0.
    expected['environment']['rewards']['knee_limit']['weight']=0.
    # Only these kernels implement the reward-version change and resume path.
    for name in ('entry.py','huananhu_rewards.py','huananhu_runner.py'):
        expected['kernels_sha256'][name]=current['kernels_sha256'][name]
    if json.loads(json.dumps(expected))!=json.loads(json.dumps(current)):
        raise ValueError('Migration permits only knee_limit -0.5 -> 0; other training/robot changes require review')
    for name in ('actor_state_dict','critic_state_dict','optimizer_state_dict','iter'):
        if name not in saved:raise ValueError(f'Full resume is missing {name}')
    iteration=saved['iter'];effective=infos.get('huananhu_training_iteration')
    if not isinstance(iteration,int) or not isinstance(effective,int) or iteration<0 or effective<iteration+1:
        raise ValueError('Invalid completed-update counters in r11 checkpoint')
    return infos

class RecoverVecEnv(RslRlVecEnvWrapper):
    """Cache returned observations for the existing CLI and native viewer."""
    def __init__(self,**kwargs):
        num_envs=kwargs.get('num_envs',4096);device=kwargs.get('device','cuda:0')
        randomize=kwargs.get('randomize',True)
        seed=kwargs.get('seed',42);torch.manual_seed(seed)
        from .model import POLICY_JOINTS
        cfg=make_huananhu_env_cfg(play=not randomize,dynamic_modes=kwargs.get('dynamic_modes',True),
            policy_joint_order=kwargs.get('policy_joint_order',POLICY_JOINTS),
            knee_limit_weight=kwargs.get('knee_limit_weight'),
            reset_config=kwargs.get('reset_config'),height_gate=kwargs.get('height_gate'),
            command_config=kwargs.get('command_config'))
        cfg.scene.num_envs=num_envs;cfg.seed=seed
        cfg.auto_reset=kwargs.get('auto_reset',True)
        super().__init__(ManagerBasedRlEnv(cfg,device=device),clip_actions=None)
        self.obs=self.get_observations()
    def step(self,actions):
        self.obs,reward,done,extra=super().step(actions)
        return self.obs,reward,done,extra

def runner_class():
    from .huananhu_runner import HuananhuOnPolicyRunner
    class LocalRecoverRunner(HuananhuOnPolicyRunner):
        def initialize_from(self,path):
            # An external profile may share dimensions but a different robot.
            # Keep the source hook, enforcing the local asset/interface first.
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved.get('infos',{}).get('interface')!=manifest():
                raise ValueError('Initialization requires an explicit local robot/interface migration')
            return super().initialize_from(path)
        def save(self,path,infos=None):
            return super().save(path,{**(infos or {}),'interface':manifest(),
                **({'reward_migration_history':self.reward_migration_history}
                   if hasattr(self,'reward_migration_history') else {})})
        def load(self,path,*args,**kwargs):
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved.get('infos',{}).get('interface')!=manifest():
                raise ValueError('r14 training/robot/command manifest mismatch; historical checkpoints require explicit migration')
            result=super().load(path,*args,**kwargs)
            self.reward_migration_history=list(saved['infos'].get('reward_migration_history',[]))
            return result
        def resume_without_knee_limit(self,path):
            from .runner import DirectMotorOnPolicyRunner
            saved=torch.load(path,map_location='cpu',weights_only=False)
            infos=validate_knee_limit_resume(saved)
            next_iteration=saved['iter']+1
            effective=infos['huananhu_training_iteration']
            digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            result=DirectMotorOnPolicyRunner.load(self,path,
                load_cfg={'actor':True,'critic':True,'optimizer':True,'iteration':True},strict=True)
            self.current_learning_iteration=next_iteration
            owner=self._progress_owner()
            owner.training_iteration=effective-self.env.unwrapped.common_step_counter//24
            self.stage_history=list(infos.get('stage_history',[]))
            if infos.get('init_source') is not None:self.init_source=infos['init_source']
            self.reward_migration_history=[*infos.get('reward_migration_history',[]),dict(
                source_checkpoint=str(Path(path).resolve()),sha256=digest,
                from_interface=infos['interface']['interface'],to_interface=INTERFACE,
                reward='knee_limit',from_weight=-.5,to_weight=0.,
                completed_updates=effective,next_iteration=next_iteration,
                actor_restored=True,critic_restored=True,optimizer_restored=True,
                learning_rate=self.alg.learning_rate)]
            print(f'[INFO] Resume r11 -> r12: knee_limit -0.5 -> 0; next iteration '
                  f'{next_iteration}, completed updates {effective}, LR {self.alg.learning_rate}',flush=True)
            return result
        def resume_with_wide_height_gate(self,path):
            from .runner import DirectMotorOnPolicyRunner
            saved = torch.load(path, map_location='cpu', weights_only=False)
            infos = validate_wide_height_gate_resume(saved)
            effective = infos['huananhu_training_iteration']
            next_iteration = saved['iter'] + 1
            result = DirectMotorOnPolicyRunner.load(self,path,
                load_cfg={'actor':True,'critic':True,'optimizer':True,'iteration':True},strict=True)
            self.current_learning_iteration = next_iteration
            self._progress_owner().training_iteration = effective - self.env.unwrapped.common_step_counter // 24
            self.stage_history = list(infos.get('stage_history', []))
            if infos.get('init_source') is not None:
                self.init_source = infos['init_source']
            self.reward_migration_history = [*infos.get('reward_migration_history', []), dict(
                source_checkpoint=str(Path(path).resolve()),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                from_interface=infos['interface']['interface'],to_interface=INTERFACE,
                changes=[dict(reward='velocity_height_gate',from_value=[.03,.06],to_value=[.05,.10]),
                         dict(reward='knee_limit',from_weight=-.5,to_weight=0.)],
                completed_updates=effective,next_iteration=next_iteration,
                actor_restored=True,critic_restored=True,optimizer_restored=True,
                learning_rate=self.alg.learning_rate)]
            print(f'[INFO] Resume r11 -> r13: height gate 3–6 -> 5–10 cm, knee_limit -0.5 -> 0; '
                  f'next iteration {next_iteration}, completed updates {effective}, LR {self.alg.learning_rate}',flush=True)
            return result
    return LocalRecoverRunner
