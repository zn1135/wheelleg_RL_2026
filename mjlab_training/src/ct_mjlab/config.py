"""Current r14 training factories; r5 constants below support historical replay.

The current configuration lives in reference_training/huananhu_env_cfg.py.
Legacy constants must not be used to configure new training.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT / 'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
RECIPE = json.loads(Path(__file__).with_name('scut_v14.json').read_text())
PARAMS = RECIPE['parameters']
WEIGHTS = {k: v for k, v in RECIPE['rewards'].items() if v != 0}
JOINTS = ('lf0', 'lf00', 'rf0', 'rf00', 'lfwheel', 'rfwheel')
INTERFACE = 'chuanliantui-mjlab-huananhu-v14-flat-standup-r5'
DT, DECIMATION, HEIGHT = .002, 10, .22
KP = (60., 60., 60., 60., 0., 0.)
KD = (2., 2., 2., 2., .1, .1)
# MJCF motor coordinates; physical angles are q - ref.
_JOINT_REFS = {j.attrib['name']: float(j.attrib.get('ref', 0))
               for j in ET.parse(ASSET).iter('joint') if 'name' in j.attrib}
LEG_REFERENCE = tuple(_JOINT_REFS[name] for name in JOINTS[:4])
LEG_TARGET_LIMIT = 3.14
STEP_DT = DT * DECIMATION
MODE_PARAMS = dict(stable_hold_s=.5, stable_tilt_deg=15., height_tolerance_m=.03,
                   wheel_contact_n=5., base_contact_n=1., failure_hold_s=.1,
                   failure_tilt_deg=40., failure_height_m=.16)

def legacy_manifest():
    return dict(interface=INTERFACE, source_commit=RECIPE['commit'], source_task=RECIPE['task'],
                asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(), joints=JOINTS,
                actor_dim=35, critic_dim=78, action_dim=6, physics_dt=DT, decimation=DECIMATION,
                height=HEIGHT, modes=dict(layout=['normal','stair','slope','recover','jump','height_target','state_time'],
                    semantics='normal-recover-standup', parameters=MODE_PARAMS,
                    height_target='recover: height command * 5; normal: 0',
                    state_time='recover elapsed seconds; normal: 0',
                    law_sha256=hashlib.sha256(Path(__file__).with_name('modes.py').read_bytes()).hexdigest()),
                reset=dict(mode='v14-normal-ground',ground_probability=.5,
                    ground_positive_per_leg_probability=.6,ground_root_height_m=.25,
                    normal_root_height_m=.35,normal_roll_pitch_rad=[-.15,.15],normal_yaw_rad=[-3.14,3.14],
                    leg_speed_rad_s=[-1.5707963267948966,1.5707963267948966],wheel_speed_rad_s=[-50,50],
                    bank_sha256=hashlib.sha256(Path(__file__).with_name('reset_poses.npz').read_bytes()).hexdigest(),
                    sampler_sha256=hashlib.sha256(Path(__file__).with_name('reset.py').read_bytes()).hexdigest()),
                gas_spring=dict(model='physical-hinge-slide',preload_m=.01,
                    reference_compression_m=.09,force_limit_n=150.,damping_ns_per_m=50.,
                    law_sha256=hashlib.sha256(Path(__file__).with_name('gas_spring.py').read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256((ASSET.parent/'spring-source.json').read_bytes()).hexdigest(),
                    mesh_sha256={name:hashlib.sha256((ASSET.parent.parent/'meshes'/name).read_bytes()).hexdigest()
                        for name in ('lf0_physical.STL','rf0_physical.STL')}), recipe_sha256=hashlib.sha256(
                    Path(__file__).with_name('scut_v14.json').read_bytes()).hexdigest(),
                poses_sha256=hashlib.sha256(Path(__file__).with_name('poses.json').read_bytes()).hexdigest(),
                control=dict(leg_scale=.5,wheel_scale=10,wheel_speed_limit=20,
                    leg_target_limit_relative_to_ref=[-LEG_TARGET_LIMIT,LEG_TARGET_LIMIT],
                    leg_reference=LEG_REFERENCE, kp=KP,kd=KD,effort=[40,40,40,40,3.9,3.9],
                    effort_scale_before_clip=True,leg_effort_scale_sampling=[.8,1.1],
                    wheel_effort_scale_sampling=[.9,1.1],
                    zero_torque_s=.2,obs_delay_sampling=[1,4],act_delay_sampling=[1,3],
                    delay_sampling_upper_exclusive=True,delay_step_s=DT,clear_delay_on_reset=True))

def legacy_runner_cfg(seed=42):
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


def manifest():
    from .reference_training.entry import manifest as current
    return current()


def runner_cfg(seed=42):
    from .reference_training.entry import runner_cfg as current
    return current(seed)
