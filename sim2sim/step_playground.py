"""Different-height single platforms for native MuJoCo interaction."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'artifacts/fudan_ppo_step/20261117-v7-final3200-01/training/model_final.pt'


class ManualStepPolicies:
    """Select compatible walk/step Actors without resetting their shared controller."""
    def __init__(self, walk_checkpoint, step_checkpoint, step_policy, step_metadata, *, start_step=False):
        from dataclasses import replace
        from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record
        from ct_mjlab.fudan_ppo_walk.policy import load_policy
        walk_policy, walk_metadata = load_policy(walk_checkpoint)
        self.training_disturbances = dict(walk=walk_metadata['disturbances'], step=step_metadata['disturbances'])
        # 与坡道交互相同：不施加训练外力/推扰，保留原动作和传感延迟。
        conditions = replace(from_record(walk_metadata['disturbances']), pushes=False,
            external_force_range=(0., 0.), external_torque_range=(0., 0.))
        walk_metadata = dict(walk_metadata, disturbances=conditions.record())
        keys = ('contract', 'reset_contract', 'reference', 'control', 'control_timing',
                'physical_wheel_model', 'wheel_response', 'disturbances')
        for key in keys:
            # JSON统一list/tuple表示；参数值、物理时序及历史语义必须相同。
            if json.dumps(walk_metadata[key], sort_keys=True) != json.dumps(step_metadata[key], sort_keys=True):
                raise ValueError(f'Walk/step physical interface differs: {key}')
        self.policies = dict(walk=walk_policy, step=step_policy)
        self.metadata = dict(walk=walk_metadata, step=step_metadata)
        self.checkpoints = {role: dict(checkpoint=str(Path(path).resolve()),
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), iterations=self.metadata[role]['iterations'])
            for role, path in [('walk', walk_checkpoint), ('step', step_checkpoint)]}
        self.mode = self.initial_mode = 'step' if start_step else 'walk'
        self.walk_height = .26
        self.transitions = []

    @property
    def height_max(self):
        return .35 if self.mode == 'step' else .30

    @property
    def command_limits(self):
        return (2.5, 2.) if self.mode == 'step' else (3., 10.)

    @property
    def keyboard_commands(self):
        return (1., .5) if self.mode == 'step' else (3., 2.)

    def select(self, mode, env):
        if mode == self.mode:
            return False
        if mode not in self.policies:
            raise ValueError('Expected walk or step policy')
        previous = self.mode
        if mode == 'step':
            self.walk_height = float(np.clip(env.command[2], .22, .30))
            env.command[2] = .35
        else:
            env.command[2] = self.walk_height
        # 不复位物理、PD、上一动作、FIFO或125维历史；下一个策略拍直接选新Actor。
        self.mode = mode
        self.transitions.append(dict(time_s=float(env.data.time), before=previous, after=mode,
                                     height_command_m=float(env.command[2])))
        return True

    def act_inference(self, observation, history):
        return self.policies[self.mode].act_inference(observation, history)

    def record(self):
        return dict(initial_mode=self.initial_mode, active_mode=self.mode,
                    checkpoints=self.checkpoints, transitions=self.transitions,
                    training_disturbances=self.training_disturbances,
                    playback_disturbances={role: value['disturbances'] for role, value in self.metadata.items()})


def layout(metadata):
    step = metadata['training_recipe']['step']
    heights = [h for h in step['heights'] if h > 0]
    if len(heights) > 9:
        raise ValueError('Native platform selection supports up to nine heights')
    result = [dict(key='0', height=0., offset=(-25., -2., 0.), label='0: flat')]
    for i, height in enumerate(heights):
        result.append(dict(key=str(i + 1), height=height,
            offset=(10. * (i % 3) - 15., 4. * (i // 3) - 6., 0.),
            label=f'{i + 1}: {height * 1000:g} mm'))
    for item in result:
        x, y, _ = item['offset']
        item['spawn'] = (x + (4. - step['approach_distance'][1] if item['height'] else 5.), y + 2.)
        item['edge'] = (x + 4., y + 2.) if item['height'] else None
    return result


def build_model(metadata):
    from ct_mjlab.envs.fudan_ppo_step.terrain import PlatformCfg
    from ct_mjlab.envs.fudan_ppo_walk.wheel_physics import mechanical_spec
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import make_env_cfg
    step = metadata['training_recipe']['step']
    spec = mechanical_spec()
    spec.worldbody.add_body(name='terrain')
    # 直接使用训练的盒体生成器，尺寸、侧障碍和垂直沿保持一致。
    for item in layout(metadata):
        level = step['heights'].index(item['height'])
        config = PlatformCfg(mode='up' if level else 'flat', size=tuple(step['patch_size']),
            heights=tuple(step['heights']), width=step['width'], depth=step['depth'],
            side_obstacle_level=step['side_obstacle_level'],
            side_obstacle_height=step['side_obstacle_height'],
            approach_distance=tuple(step['approach_distance']))
        output = config.function(level / (len(step['heights']) - 1), spec, np.random.default_rng(1))
        for index, geometry in enumerate(output.geometries):
            geom = geometry.geom
            geom.name = f'step_{item["key"]}_{index}'
            geom.pos = np.asarray(geom.pos) + item['offset']
            geom.rgba = ((.22, .27, .31, 1.) if index == 0 else
                         (.25 + item['height'], .62 - item['height'], .35, 1.) if index == 1 else
                         (.42, .43, .47, 1.))
    model = spec.compile()
    make_env_cfg(play=True).sim.mujoco.apply(model)
    model.opt.timestep = metadata['control_timing']['physics_dt']
    model.geom_friction[:, 0] = .8
    terrain_id = model.body('terrain').id
    model.geom_priority[model.geom_bodyid != terrain_id] = 1
    model.vis.headlight.ambient[:] = .4
    model.vis.headlight.diffuse[:] = .5
    return model
