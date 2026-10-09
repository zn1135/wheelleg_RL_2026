"""SCUT V14 flat v0 recipe; robot and standup task overrides are explicit."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT / 'sim2sim/chuanliantui.xml'
RECIPE = json.loads(Path(__file__).with_name('scut_v14.json').read_text())
PARAMS = RECIPE['parameters']
WEIGHTS = {k: v for k, v in RECIPE['rewards'].items() if v != 0}
JOINTS = ('lf0', 'lf00', 'rf0', 'rf00', 'lfwheel', 'rfwheel')
INTERFACE = 'chuanliantui-mjlab-huananhu-v14-flat-standup-r1'
DT, DECIMATION, HEIGHT = .005, 4, .22
STEP_DT = DT * DECIMATION

def manifest():
    return dict(interface=INTERFACE, source_commit=RECIPE['commit'], source_task=RECIPE['task'],
                asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(), joints=JOINTS,
                actor_dim=35, critic_dim=78, action_dim=6, physics_dt=DT, decimation=DECIMATION,
                height=HEIGHT, gas_spring_force_n=150, recipe_sha256=hashlib.sha256(
                    Path(__file__).with_name('scut_v14.json').read_bytes()).hexdigest(),
                poses_sha256=hashlib.sha256(Path(__file__).with_name('poses.json').read_bytes()).hexdigest(),
                control=dict(leg_scale=.5,wheel_scale=10,wheel_speed_limit=20,
                    leg_target_limit=[-3.141592653589793,3.141592653589793],
                    kp=[10,10,10,10,0,0],kd=[1,1,1,1,.1,.1],effort=[40,40,40,40,3.9,3.9],
                    effort_scale_before_clip=True,leg_effort_scale_sampling=[.8,1.1],
                    wheel_effort_scale_sampling=[.9,1.1],
                    zero_torque_s=.2,obs_delay_sampling=[1,4],act_delay_sampling=[1,3],
                    delay_sampling_upper_exclusive=True,delay_step_s=DT,clear_delay_on_reset=True))

def runner_cfg(seed=42):
    from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg
    cfg=RslRlOnPolicyRunnerCfg(seed=seed, num_steps_per_env=24, max_iterations=20000,
        save_interval=500, experiment_name='chuanliantui', logger='tensorboard', upload_model=False,
        actor=RslRlModelCfg(hidden_dims=(256,128,64), activation='elu', obs_normalization=False,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(256,128,64), activation='elu', obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=4., learning_rate=1e-4, entropy_coef=.005,
            num_learning_epochs=5, num_mini_batches=4, clip_param=.2, gamma=.99, lam=.95,
            schedule='adaptive', desired_kl=.01, max_grad_norm=1., use_clipped_value_loss=True))
    result=asdict(cfg)
    # Match mjlab's MjlabOnPolicyRunner adaptation to RSL-RL MLPModel.
    for key in ('actor','critic'):
        for optional in ('cnn_cfg','rnn_type','rnn_hidden_dim','rnn_num_layers'):
            result[key].pop(optional,None)
        if result[key]['distribution_cfg'] is None:
            result[key].pop('distribution_cfg')
    return result
