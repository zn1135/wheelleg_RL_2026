"""RMUC2027 full-field native MuJoCo playground; not a GPU acceptance entry."""
import argparse
from collections import deque
from dataclasses import replace
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import DisturbanceState, SENSOR_SLOTS, from_record
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.reset_geometry import ResetGeometry
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.wheel_physics import mechanical_spec, record as wheel_record
from ct_mjlab.fudan_ppo_walk.contract import MOTORS, LEGS, RESET_CONTRACT, RESET_V2_CONTRACT
from ct_mjlab.fudan_ppo_walk.policy import load_policy

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'ffd1eb7d1d1a2d1654125d8f316fdb0fdac6d001'
ARENA = ROOT / f'artifacts/rmuc2027/upstream-{REVISION}/rmuc2027_mjarena/arena_mjcf/model.xml'
COARSE_ARENA = ROOT / 'artifacts/rmuc2027/continuous50-v8/arena.xml'
CHECKPOINT = ROOT / 'logs/mjlab/fudan_ppo_walk/20261114_pd200_p20d1_half_random_delay_wide_resume1000_to3000_01/model_final.pt'
REGIONAL_CHECKPOINT = ROOT / 'artifacts/fudan_ppo_walk/models/20261116_model6838_regions_scan_slope5to20_mu08/model_final.pt'
WHEELS = (2, 5)


def regional_metadata(saved):
    # 交互体验关闭外载，训练记录和两条延迟队列保持原值。
    conditions = replace(from_record(saved['disturbances']), pushes=False,
                         external_force_range=(0., 0.), external_torque_range=(0., 0.))
    return dict(saved, disturbances=conditions.record())


def build_regional_model(metadata):
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.terrain import RegionalTerrainCfg, VERSION, record
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import make_env_cfg
    terrain = metadata['terrain']
    if terrain['version'] != VERSION or terrain['mode'] != 'regions':
        raise ValueError('Regional playback requires a regional checkpoint')
    task = FudanWalkCfg()
    task.terrain = SimpleNamespace(**{key: terrain[key] for key in vars(task.terrain)})
    record(task)
    spec = mechanical_spec()
    # 生成器局部坐标为0～size，训练场地的世界中心为(0,0)。
    spec.worldbody.add_body(name='terrain', pos=(-task.terrain.size/2, -task.terrain.size/2, 0.))
    RegionalTerrainCfg(terrain=task.terrain).function(0., spec, np.random.default_rng(task.terrain.seed))
    model = spec.compile()
    make_env_cfg(play=True).sim.mujoco.apply(model)
    model.opt.timestep = metadata['control_timing']['physics_dt']
    model.geom_friction[:, 0] = .8
    terrain_body = model.body('terrain').id
    model.geom_priority[model.geom_bodyid != terrain_body] = 1
    # 只改显示颜色和光照，让坡面与崎岖分区容易辨认。
    for name, color in (('walk_flat', (.25, .4, .55, 1.)),
                        ('walk_pyramid_base', (.4, .5, .3, 1.)),
                        ('walk_rough', (.65, .42, .24, 1.)),
                        ('walk_outer_floor', (.12, .15, .18, 1.))):
        model.geom_rgba[model.geom(name).id] = color
    model.vis.headlight.ambient[:] = .35
    model.vis.headlight.diffuse[:] = .45
    model.light_dir[0] = [-1., 1., -.2]
    return model


def build_model(metadata, arena=ARENA):
    spec = mechanical_spec()
    if arena is not None:
        if not arena.is_file():
            raise FileNotFoundError('请先运行 scripts/setup_rmuc2027.sh')
        print('Loading complete RMUC2027 field (first compilation can take several minutes)...', flush=True)
        field = mujoco.MjSpec.from_file(str(arena))
        spec.attach(field, prefix='arena/', frame=spec.worldbody.add_frame())
    else:
        spec.worldbody.add_geom(name='terrain', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=(20., 20., .1), friction=(1., .005, .0001))
    model = spec.compile()
    # 场地不接管机器人的积分、求解器或控制周期。
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import make_env_cfg
    make_env_cfg(play=True).sim.mujoco.apply(model)
    model.opt.timestep = metadata['control_timing']['physics_dt']
    for i in range(model.ngeom):
        if not model.geom(i).name.startswith('arena/') and model.geom(i).name != 'terrain':
            model.geom_priority[i] = 1
    return model


def model_cache(metadata, arena):
    import xml.etree.ElementTree as ET
    arena = Path(arena).resolve()
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import make_env_cfg
    digest = hashlib.sha256(mujoco.__version__.encode() + inspect.getsource(build_model).encode())
    digest.update(repr(make_env_cfg(play=True).sim.mujoco).encode())
    paths = [MODEL_XML, MODEL_XML.with_name('wheel_mechanical.json'),
             *sorted((MODEL_XML.parent.parent / 'meshes').rglob('*')), *sorted(arena.parent.rglob('*'))]
    if arena.is_file():
        paths.extend(sorted({(arena.parent / node.attrib['file']).resolve()
            for node in ET.parse(arena).iter()
            if node.tag in ('mesh', 'texture', 'hfield') and 'file' in node.attrib}))
    for path in paths:
        if path.is_file():
            digest.update(str(path).encode()); digest.update(path.read_bytes())
    digest.update(json.dumps(metadata['control_timing'], sort_keys=True).encode())
    return ROOT / 'artifacts/rmuc2027/cache' / (digest.hexdigest() + '.mjb')


def configure_optimized_field(model):
    # 附加场地时父模型的全局显示选项优先。关闭仅供调试显示用的包围盒标记。
    model.vis.global_.bvactive = False
    # 静态场地的位姿只计算一次，活动机构每步照常积分，不进入休眠。
    model.tree_sleep_policy[:] = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP

    fences = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'arena/rmuc2027_fence_{side}')
              for side in ('east', 'west', 'north', 'south')]
    fences = [index for index in fences if index >= 0]
    if fences:
        # 围栏需要阻挡机器人，但较软的法向接触与低摩擦可减少撞栏时闭链机构受冲击。
        model.geom_solref[fences] = (.04, 1.4)
        model.geom_solimp[fences] = (.75, .90, .003, .5, 2.)
        model.geom_friction[fences] = (.12, .001, .0001)
        model.geom_priority[fences] = 2


class ArenaPlayback:
    def __init__(self, model, metadata, seed=1):
        if metadata['reset_contract'] not in (RESET_CONTRACT, RESET_V2_CONTRACT):
            raise ValueError('Only PD-centred v2/v3 checkpoints are supported')
        if metadata['physical_wheel_model'] != wheel_record('mechanical'):
            raise ValueError('Checkpoint mechanical wheel parameters differ from the current robot')
        response = metadata['wheel_response']
        if (response['gain'], response['delay_ms'], response['time_constant_ms'],
                response.get('time_constant_range_ms'), response['torque_limit_nm']) != (1., 0., 0., None, None):
            raise ValueError('This playground requires unmodified mechanical wheel response')
        self.conditions = from_record(metadata['disturbances'])
        if self.conditions.pushes or self.conditions.external_force_range != (0., 0.) or self.conditions.external_torque_range != (0., 0.):
            raise ValueError('This interactive playground supports checkpoints without external disturbances')
        self.metadata, self.model = metadata, model
        self.data = mujoco.MjData(model)
        self.control = SimpleNamespace(**metadata['control'])
        timing = metadata['control_timing']
        self.dt, self.policy_dt = timing['physics_dt'], timing['policy_dt']
        self.decimation = self.control.decimation
        self.pd_decimation = self.control.pd_decimation
        if not (np.isclose(self.dt * self.decimation, self.policy_dt) and
                np.isclose(self.dt * self.pd_decimation, timing['pd_dt'])):
            raise ValueError('Checkpoint control timing is inconsistent')
        self.q = np.array([model.joint(n).qposadr[0] for n in MOTORS])
        self.v = np.array([model.joint(n).dofadr[0] for n in MOTORS])
        self.motor_ids = np.array([model.actuator(n + '_motor').id for n in MOTORS])
        self.spring_ids = np.array([model.actuator(s + '_gas_spring_motor').id for s in ('left', 'right')])
        joints = model.actuator_trnid[self.spring_ids, 0]
        self.spring_q, self.spring_v = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.base = model.body('base_link').id
        self.reference = model.qpos0[self.q].astype(np.float32)
        self.nominal = self.reference.copy()
        self.nominal[list(LEGS)] = self.control.nominal_leg_positions
        self.pose = ResetGeometry().solve(self.nominal)
        self.normalization = FudanWalkCfg.normalization
        self.command = np.array([0., 0., .26], dtype=np.float32)
        torch.manual_seed(seed)
        self.delays = DisturbanceState(self.conditions, 1, 'cpu', seed, physics_dt=self.dt)
        self.last_action = np.zeros(6, dtype=np.float32)
        self.velocity = np.zeros(6, dtype=np.float32)
        self.torque = np.zeros(6, dtype=np.float32)
        self.spawn = None

    def gravity(self):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        return rotation.reshape(3, 3).T @ np.array([0., 0., -1.])

    def sensors(self):
        n = self.normalization
        return np.concatenate((self.data.qvel[3:6] * n.angular_velocity, self.gravity(),
            (self.data.qpos[self.q] - self.nominal)[list(LEGS)], self.velocity * n.joint_velocity)).astype(np.float32)

    def _raw_observation(self, scale):
        sensor = self.sensors()
        return np.concatenate((sensor[:6], self.command * scale,
                                sensor[6:], self.last_action)).astype(np.float32)

    def direct_observation(self, scale):
        return self._raw_observation(scale).clip(-self.normalization.clip, self.normalization.clip)

    def observation(self):
        frame = self._raw_observation(self.normalization.command)
        if self.conditions.sensor_delay:
            frame[list(SENSOR_SLOTS)] = self.delays.observed()[0].numpy()
        self.history_frame = frame.copy()
        return frame.clip(-self.normalization.clip, self.normalization.clip)

    def reset(self, spawn, yaw=None):
        self.spawn = np.asarray(spawn, dtype=float)
        if yaw is not None:
            self.spawn_yaw = float(yaw)
        heading = getattr(self, 'spawn_yaw', 0.)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.spawn + [0., 0., self.pose['height']]
        self.data.qpos[3:7] = (np.cos(heading/2), 0., 0., np.sin(heading/2))
        for name, position in self.pose['joint_positions'].items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = position
        self.last_action.fill(0.)
        self.velocity.fill(0.)
        self.torque.fill(0.)
        self.pd_tick = 0
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        self.last_sample_q = self.data.qpos[self.q].copy()
        self.delays.reset(slice(None), torch.from_numpy(self.sensors()[None]))
        # 与现有v3保持一致：零动作热步后，带延迟的首帧填满五格历史。
        clean = self.observation()
        self.history = np.tile(clean, 5)
        obs = self.step(np.zeros(6, dtype=np.float32))
        if self.metadata['reset_contract'] == RESET_CONTRACT:
            self.history = np.tile(obs, 5)

    def apply_controls(self, action, *, direct=False):
        applied = self.delays.action(torch.from_numpy(action[None]))[0].numpy()
        # Jump 直通，同时把实际发出的新动作写进行走队列，回切时队列持续有效。
        if direct:
            applied = action
        if self.pd_tick == 0:
            c = self.control
            q, v = self.data.qpos[self.q], self.data.qvel[self.v]
            legs, wheels = list(LEGS), list(WHEELS)
            target = np.clip(self.nominal[legs] + c.leg_position_scale * applied[legs],
                             self.reference[legs] - 3.14, self.reference[legs] + 3.14)
            self.torque[legs] = c.leg_stiffness * (target - q[legs]) - c.leg_damping * v[legs]
            speed = np.clip(c.wheel_velocity_scale * applied[wheels], -c.max_wheel_velocity, c.max_wheel_velocity)
            self.torque[wheels] = c.wheel_stiffness * (speed - v[wheels]) - c.wheel_damping * v[wheels]
            self.torque[:] = self.torque.clip([-40., -40., -3.9, -40., -40., -3.9], [40., 40., 3.9, 40., 40., 3.9])
        self.pd_tick = (self.pd_tick + 1) % self.pd_decimation
        self.data.ctrl[self.motor_ids] = self.torque
        self.data.ctrl[self.spring_ids] = gas_spring_force(self.data.qpos[self.spring_q], self.data.qvel[self.spring_v])

    def step(self, action, *, direct=False, sample=None):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError('Policy must output six finite actions')
        self.last_action[:] = action.clip(-self.normalization.clip, self.normalization.clip)
        apply = partial(self.apply_controls, direct=True) if direct else self.apply_controls
        for _ in range(self.decimation):
            apply(self.last_action)
            mujoco.mj_step(self.model, self.data)
            q = self.data.qpos[self.q].copy()
            self.velocity[:] = ((q - self.last_sample_q + np.pi) % (2 * np.pi) - np.pi) / self.dt
            self.last_sample_q = q
            self.delays.sample(torch.from_numpy(self.sensors()[None]))
            if sample is not None:
                sample()
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.warning.number.any():
            raise FloatingPointError('Native MuJoCo reported a non-finite state or simulation warning')
        obs = self.observation()
        frame = obs if self.metadata['reset_contract'] == RESET_CONTRACT else self.history_frame
        self.history = np.concatenate((self.history[25:], frame))
        return obs


def ground_height(model, x, y, regional=False):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # 只向场地的碰撞组投射，避开视觉外壳与机器人。
    groups = np.array([1, 0, 0, 0, 0, 0] if regional else [0, 0, 0, 1, 0, 0], dtype=np.uint8)
    # mj_ray忽略透明几何；仅在查询期间显露场地碰撞壳，随后还原显示属性。
    alpha = model.geom_rgba[:, 3].copy()
    field = [i for i in range(model.ngeom) if (model.geom_bodyid[i] == model.body('terrain').id
             if regional else model.geom(i).name.startswith('arena/'))]
    try:
        model.geom_rgba[:, 3] = 0.
        model.geom_rgba[field, 3] = 1.
        distance = mujoco.mj_ray(model, data, np.array([x, y, 5.]), np.array([0., 0., -1.]), groups, True, -1, None)
    finally:
        model.geom_rgba[:, 3] = alpha
    if distance < 0:
        raise ValueError('Spawn point is outside the field collision surface')
    return 5. - distance


def command_schedule(path, policy_dt, limits=(2., 2.), height_max=.30, events=None):
    """分段命令按策略拍对齐，避免浮点边界重复或漏掉停车命令。"""
    if path is None:
        return []
    items = json.loads(Path(path).read_text())
    if not isinstance(items, list) or not items:
        raise ValueError('Command sequence must be a nonempty JSON list')
    result, end = [], 0
    for item in items:
        seconds = float(item['duration_s'])
        command = np.asarray(item['command'], dtype=float)
        ticks = round(seconds/policy_dt)
        if (not np.isfinite(seconds) or ticks <= 0 or not np.isclose(ticks*policy_dt, seconds)
                or command.shape != (3,) or not np.isfinite(command).all()
                or not ((command >= [-limits[0], -limits[1], .22]) & (command <= [*limits, height_max])).all()):
            raise ValueError('Invalid command segment duration or command bounds')
        if "key" in item:
            if events is None or item["key"] not in ("T", "SPACE", "J", "G", "R", "X"):
                raise ValueError("Unsupported key in command sequence")
            events[end] = "SPACE" if item["key"] == "J" else item["key"]
        end += ticks
        result.append((end, command.tolist()))
    return result


def scheduled_command(schedule, tick):
    for end, command in schedule:
        if tick < end:
            return command
    return schedule[-1][1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--walk-checkpoint', type=Path, help='Steps mode: walking Actor before pressing T')
    parser.add_argument('--parking-checkpoint', type=Path, help='Steps mode: use this Actor when stopping after forward motion')
    parser.add_argument('--step-mode', action='store_true', help='Steps mode: start with the stair Actor, equivalent to pressing T')
    parser.add_argument('--terrain', choices=('arena', 'regions', 'steps'), default='arena')
    parser.add_argument('--arena', type=Path, help='Explicit field XML; defaults to RMUC2027 without the central fixtures')
    parser.add_argument('--viewer', choices=('native', 'viser', 'none'), default='native')
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--spawn', nargs=2, type=float, metavar=('X', 'Y'))
    parser.add_argument('--spawn-yaw-deg', type=float, default=0., help='World heading; red platform approach is -90')
    parser.add_argument('--command-sequence', type=Path, help='JSON duration_s/command list; requires --viewer none')
    parser.add_argument('--keyboard-speed', type=float, default=None, help='Override the active policy keyboard speed')
    parser.add_argument('--keyboard-yaw', type=float, default=None)
    parser.add_argument('--spin', action='store_true', help='Start walking spin mode: vx=0, yaw=+10rad/s')
    parser.add_argument('--command', nargs=3, type=float, metavar=('VX', 'YAW', 'HEIGHT'))
    parser.add_argument('--seconds', type=float, help='Stop after this many simulated seconds')
    parser.add_argument('--output', type=Path, help='New directory for trace and run report')
    args = parser.parse_args()
    regional = args.terrain == 'regions'
    steps = args.terrain == 'steps'
    generated = regional or steps
    arena_multi = args.terrain == "arena" and args.checkpoint is None
    switching = steps or arena_multi
    limits = (2.5, 2.) if steps and args.step_mode else (3., 10.) if generated or arena_multi else (2., 2.)
    height_max = .35 if steps and args.step_mode else .30
    from sim2sim.step_playground import (CHECKPOINT as STEP_CHECKPOINT, layout as step_layout,
        build_model as build_step_model, ManualStepPolicies)
    args.checkpoint = args.checkpoint or (STEP_CHECKPOINT if steps else REGIONAL_CHECKPOINT if regional or arena_multi else CHECKPOINT)
    args.command = args.command or (0., 0., .35 if steps and args.step_mode else .26)
    if not steps and (args.walk_checkpoint is not None or args.step_mode or args.parking_checkpoint is not None):
        parser.error('--walk-checkpoint, --parking-checkpoint and --step-mode require --terrain steps')
    if not steps:
        args.spawn = args.spawn or ((8.25, 4.) if regional else (-5., -3.))
    if args.command_sequence is not None and args.viewer != 'none':
        parser.error('--command-sequence requires --viewer none')
    if args.spin and (not (generated or arena_multi) or steps and args.step_mode or args.command_sequence is not None):
        parser.error('--spin requires regional or steps WALK mode and no command sequence')
    if not np.isfinite(args.spawn_yaw_deg):
        parser.error('Spawn heading must be finite')
    if args.keyboard_speed is not None and not 0 < args.keyboard_speed <= limits[0]:
        parser.error(f'Keyboard speed must be in(0,{limits[0]}]')
    if args.keyboard_yaw is not None and not 0 < args.keyboard_yaw <= limits[1]:
        parser.error(f'Keyboard yaw must be in(0,{limits[1]}]')
    if not np.isfinite(args.command).all() or not ((np.array(args.command) >= [-limits[0], -limits[1], .22])
            & (np.array(args.command) <= [*limits, height_max])).all():
        parser.error('Command is outside the playback range')
    if args.viewer == 'none' and args.command_sequence is None and (args.seconds is None or args.seconds <= 0):
        parser.error('--viewer none requires --seconds > 0')
    if args.seconds is not None and args.seconds <= 0:
        parser.error('--seconds must be positive')
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    policy_switch = None
    if arena_multi:
        from sim2sim.arena_policies import ArenaPolicies, FieldContacts
        policy = policy_switch = ArenaPolicies(args.checkpoint)
        saved = policy_switch.metadata['walk']
    else:
        policy, saved = load_policy(args.checkpoint)
    if steps:
        policy_switch = ManualStepPolicies(args.walk_checkpoint or REGIONAL_CHECKPOINT, args.checkpoint,
            policy, saved, start_step=args.step_mode, parking_checkpoint=args.parking_checkpoint)
        policy = policy_switch
    metadata = regional_metadata(saved) if regional else saved
    platforms = step_layout(metadata) if steps else []
    platform_keys = '\n'.join(' | '.join(p['label'] for p in platforms[i:i+6])
                             for i in range(0, len(platforms), 6))
    selected_platform = next((p for p in platforms if np.isclose(p['height'], .20)), None)
    if steps:
        selected_platform = selected_platform or platforms[-1]
        args.spawn = args.spawn or selected_platform['spawn']
    terrain_record = (dict(mode='steps', platforms=platforms) if steps else
                      saved.get('terrain') if regional else None)
    scheduled_keys = {}
    schedule = command_schedule(args.command_sequence, metadata['control_timing']['policy_dt'], limits,
                                .35 if arena_multi else height_max, scheduled_keys if arena_multi else None)
    if schedule:
        duration = schedule[-1][0]*metadata['control_timing']['policy_dt']
        if args.seconds is not None and not np.isclose(args.seconds, duration):
            parser.error('--seconds must equal the command sequence duration, or be omitted')
        args.seconds = duration
    keyboard_speed = args.keyboard_speed if args.keyboard_speed is not None else (
        policy_switch.keyboard_commands[0] if switching else .5 if metadata.get('task') == 'fudan_ppo_step' else (3. if regional else 2.))
    keyboard_yaw = args.keyboard_yaw if args.keyboard_yaw is not None else (
        policy_switch.keyboard_commands[1] if switching else .5 if metadata.get('task') == 'fudan_ppo_step' else 2.)
    without_rune = args.terrain == 'arena' and args.arena is None
    if without_rune:
        from sim2sim.rmuc2027_field import default_field
        from sim2sim.rmuc2027_flat_collision import optimized_field
        args.arena = optimized_field(default_field(ARENA))
    # 二进制缓存只加速下次启动；场地/机器人资产或编译配置变化会换缓存键。
    if steps:
        model = build_step_model(metadata)
    elif regional:
        model = build_regional_model(metadata)
    else:
        cache = model_cache(metadata, args.arena)
        if cache.is_file():
            model = mujoco.MjModel.from_binary_path(str(cache))
        else:
            model = build_model(metadata, args.arena)
            cache.parent.mkdir(parents=True, exist_ok=True)
            # 其他启动进程只能读到已完整写入的编译缓存。
            with NamedTemporaryFile(dir=cache.parent, suffix='.mjb', delete=False) as stream:
                pending = Path(stream.name)
            try:
                mujoco.mj_saveModel(model, str(pending))
                pending.replace(cache)
            finally:
                pending.unlink(missing_ok=True)
    optimized = not generated and args.arena.with_name('merged.json').is_file()
    if optimized:
        configure_optimized_field(model)
    env = ArenaPlayback(model, metadata)
    env.command[:] = args.command
    if policy_switch is not None:
        env.command[2] = np.clip(env.command[2], .22, policy_switch.height_max)
    spin = args.spin
    if spin:
        env.command[:2] = (0., 10.)
    z = ground_height(model, *args.spawn, regional=generated)
    spawn = (*args.spawn, z)
    env.reset(spawn, yaw=np.deg2rad(args.spawn_yaw_deg))
    if arena_multi:
        policy_switch.reset(env)
        field_contacts = FieldContacts(model)
        samples = []
        def sample_contacts():
            samples.append(field_contacts.sample(model, env.data))
    contact_trace = []
    events, rows, poses = deque(), [], []
    paused = False
    numerical_stop = False
    # 有限回放保留异常退出，桌面手动体验允许按 R 恢复。
    allow_manual_recovery = args.viewer == 'native' and args.seconds is None
    server = handle = scene = status = None
    native_window = None
    spin_toggle = None
    step_toggle = None
    if args.viewer == 'native':
        from mujoco import viewer as native_viewer
        from mujoco.glfw import glfw
        direction_keys = {glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT}
        arena_keys = {glfw.KEY_T, glfw.KEY_SPACE, glfw.KEY_G} if arena_multi else set()
        latched_keys = set()
        def keypress(key):
            nonlocal native_window
            native_window = glfw.get_current_context()
            if key in arena_keys:
                # 回调保留两个物理步之间的短按，逐拍检查松开，过滤长按重复。
                if key not in latched_keys:
                    latched_keys.add(key)
                    events.append("SPACE" if key == glfw.KEY_SPACE else chr(key))
            elif key not in direction_keys and not (steps and key == glfw.KEY_T):
                events.append(chr(key))
        handle = native_viewer.launch_passive(model, env.data, key_callback=keypress,
            show_left_ui=not generated, show_right_ui=not generated)
        handle.cam.lookat[:] = env.data.qpos[:3]
        handle.cam.distance = 3.5 if generated else 4.
        handle.cam.azimuth = 150. if generated else 90.
        handle.cam.elevation = -25.
        # 完整场地只显示动态状态；关闭重阴影/反射，碰撞与机器人控制保持。
        with handle.lock():
            if generated or optimized:
                handle.opt.geomgroup[3] = 0
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        print(f'Hold Up/Down: +/-{keyboard_speed}m/s; Left/Right: +/-{keyboard_yaw}rad/s; release: 0; E/Q height; X stop; P pause; R reset', flush=True)
        if arena_multi:
            print('G spin | T walk/step | Up: STEP 6200 | release: PARK 5200 | Space: jump once', flush=True)
        if regional:
            print('G spin ON/OFF (vx=0,yaw=10); X stop; 1 flat; 2/3/4/5 slope 5/10/15/20deg; 6 rough; V overview/robot', flush=True)
        if steps:
            print('Platforms: ' + ' | '.join(p['label'] for p in platforms) + ' | V overview/robot', flush=True)
            print(f'T walk{policy_switch.metadata["walk"]["iterations"]} / step{policy_switch.metadata["step"]["iterations"]}; '
                  'enter step: height0.35m; X stops without leaving the current mode', flush=True)
            print('WALK: arrows +/-3m/s, +/-2rad/s; G spin10rad/s', flush=True)
            if policy_switch.parking_enabled:
                print(f'STEP: Up {policy_switch.step_keyboard_speed:g}m/s / '
                      f'{policy_switch.metadata["step"]["iterations"]}; release or X / PARK '
                      f'{policy_switch.metadata["parking"]["iterations"]}', flush=True)
            else:
                print(f'STEP: arrows +/-{policy_switch.step_keyboard_speed:g}m/s, +/-0.5rad/s', flush=True)
    elif args.viewer == 'viser':
        import viser
        from mjlab.viewer.viser.scene import MjlabViserScene
        server = viser.ViserServer(host='127.0.0.1', port=args.port,
            label='steps / WALK + STEP native playground' if steps else
                  f'{args.terrain} / Fudan{metadata["iterations"]} native playground')
        scene = MjlabViserScene(server, model, 1)
        scene.create_scene_gui(camera_distance=4., show_debug_viz_control=False)
        vx = server.gui.add_slider('前后速度 m/s', min=-limits[0], max=limits[0], step=.1, initial_value=float(env.command[0]))
        yaw = server.gui.add_slider('转向 rad/s', min=-limits[1], max=limits[1], step=.1, initial_value=float(env.command[1]))
        height = server.gui.add_slider('机身高度 m', min=.22,
            max=policy_switch.height_max if switching else height_max, step=.01, initial_value=float(env.command[2]))
        pause = server.gui.add_checkbox('暂停', initial_value=False)
        if generated or arena_multi:
            spin_toggle = server.gui.add_checkbox('小陀螺：原地10rad/s', initial_value=spin,
                disabled=switching and policy_switch.mode == 'step')
        if switching:
            step_toggle = server.gui.add_checkbox('上台阶模型（T切换）', initial_value=policy_switch.mode == 'step')
            step_toggle.on_update(lambda event: events.append('MODEL_STEP' if event.target.value else 'MODEL_WALK'))
        if arena_multi:
            jump_button = server.gui.add_button('跳一次')
            jump_button.on_click(lambda _: events.append('SPACE'))
        if steps:
            selector = server.gui.add_dropdown('台阶高度', options=tuple(p['label'] for p in platforms),
                initial_value=selected_platform['label'])
            selector.on_update(lambda event: events.append(event.target.value.split(':')[0]))
        server.gui.add_button('停车').on_click(lambda _: events.append('X'))
        server.gui.add_button('回到出生点').on_click(lambda _: events.append('R'))
        status = server.gui.add_markdown('正在初始化场地与机器人…')
        @server.on_client_connect
        def connected(client):
            client.camera.look_at = tuple(env.data.qpos[:3])
            client.camera.position = tuple(env.data.qpos[:3] + [3., -4., 2.5])
        print(f'Viewer: http://127.0.0.1:{server.get_port()}', flush=True)
    startup = dict(ngeom=model.ngeom, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        iterations=metadata['iterations'], spawn=spawn, terrain=terrain_record if generated else str(args.arena),
        friction=.8 if generated else None, training_disturbances=saved['disturbances'],
        playback_disturbances=metadata['disturbances'], timing=metadata['control_timing'],
        action_delay_steps=int(env.delays.action_steps[0]), sensor_delay_steps=int(env.delays.sensor_steps[0]),
        keyboard_speed=keyboard_speed, keyboard_yaw=keyboard_yaw, spin_initial=spin,
        scope='interactive_native_playground_not_formal_acceptance')
    if not generated:
        startup['compiled_scene_key'] = cache.stem
    if without_rune:
        startup['arena_variant'] = 'clear_centre_flat_fenced'
    if policy_switch is not None:
        current = policy_switch.checkpoints[policy_switch.mode]
        startup.update(checkpoint=current['checkpoint'], checkpoint_sha256=current['sha256'],
                       iterations=current['iterations'], policy_switch=policy_switch.record(),
                       training_disturbances=policy_switch.training_disturbances[policy_switch.mode],
                       playback_disturbances=policy_switch.metadata[policy_switch.mode]['disturbances'])
    print(json.dumps(startup), flush=True)
    if args.output is not None:
        (args.output / 'startup.json').write_text(json.dumps(startup, indent=2) + '\n')
    start = time.monotonic()
    total_steps = 0
    failure = None
    next_draw = 0.
    draw_count = 0
    overview = False
    contacts, contact_error = 0, 0.
    spin_steps = 0
    policy_roles = []
    step_key_down = False
    try:
        with torch.inference_mode():
            while args.seconds is None or total_steps < round(args.seconds / env.policy_dt):
                tick = time.monotonic()
                if handle is not None and not handle.is_running():
                    break
                if handle is not None:
                    # 回调只通知按下；逐拍读取真实按键状态，松开/失焦时该轴立即归零。
                    env.command[:2] = 0.
                    t_held = False
                    held_axes = np.zeros(2)
                    native_stop = False
                    if native_window and glfw.get_window_attrib(native_window, glfw.FOCUSED):
                        held = lambda key: int(glfw.get_key(native_window, key) == glfw.PRESS)
                        held_axes[:] = (held(glfw.KEY_UP) - held(glfw.KEY_DOWN), held(glfw.KEY_LEFT) - held(glfw.KEY_RIGHT))
                        env.command[:2] = held_axes * (keyboard_speed, keyboard_yaw)
                        native_stop = bool(held(glfw.KEY_X))
                        if native_stop:
                            env.command[:2] = 0.
                            spin = False
                        for key in tuple(latched_keys):
                            if not held(key):
                                latched_keys.discard(key)
                        if steps:
                            t_held = bool(held(glfw.KEY_T))
                    elif arena_multi:
                        latched_keys.clear()
                    if steps and t_held and not step_key_down:
                        events.append('T')
                    step_key_down = t_held
                if server is not None:
                    env.command[:] = (vx.value, yaw.value, height.value)
                    paused = pause.value
                    if spin_toggle is not None:
                        spin = spin_toggle.value
                if schedule:
                    env.command[:] = scheduled_command(schedule, total_steps)
                if total_steps in scheduled_keys:
                    events.append(scheduled_keys[total_steps])
                while events:
                    key = events.popleft()
                    if numerical_stop and key != 'R':
                        continue
                    if key == 'P': paused = not paused
                    elif key == 'R':
                        if arena_multi:
                            policy_switch.transitions.append(dict(time_s=float(env.data.time),
                                before=policy_switch.active_role, after='walk', reason='reset'))
                            env.command[2] = .26
                            policy_switch.mode = 'walk'
                            limits = policy_switch.command_limits
                            keyboard_speed = args.keyboard_speed or 3.
                            keyboard_yaw = args.keyboard_yaw or 2.
                            if server is not None:
                                step_toggle.value = False
                                spin_toggle.disabled = False
                                vx.min, vx.max = -limits[0], limits[0]
                                yaw.min, yaw.max = -limits[1], limits[1]
                                height.max, height.value = .30, .26
                        if generated or arena_multi:
                            spin = False
                            env.command[:2] = 0.
                            if spin_toggle is not None: spin_toggle.value = False
                            if server is not None: vx.value = yaw.value = 0.
                        if numerical_stop:
                            env.command[:2] = 0.
                        env.reset(spawn)
                        if numerical_stop:
                            numerical_stop = paused = False
                            print(json.dumps(dict(event='numerical_stop_reset', spawn=spawn)), flush=True)
                        if arena_multi: policy_switch.reset(env)
                        if steps: policy_switch.reset_parking()
                    elif key == 'X':
                        spin = False
                        env.command[:2] = 0.
                        if spin_toggle is not None: spin_toggle.value = False
                        if server is not None: vx.value = yaw.value = 0.
                    elif arena_multi and key == 'SPACE':
                        if not paused and policy_switch.start_jump(env):
                            if spin:
                                env.command[:2] = 0.
                            spin = False
                            if spin_toggle is not None: spin_toggle.value = False
                    elif arena_multi and policy_switch.gate is not None and key in ('G','T','MODEL_STEP','MODEL_WALK'):
                        continue
                    elif (regional or switching and policy_switch.mode == 'walk') and key == 'G':
                        spin = not spin
                        env.command[:2] = 0.
                    elif key == 'E': env.command[2] += .01
                    elif key == 'Q': env.command[2] -= .01
                    elif switching and key in ('T', 'MODEL_STEP', 'MODEL_WALK'):
                        mode = ('step' if policy_switch.mode == 'walk' else 'walk') if key == 'T' else (
                            'step' if key == 'MODEL_STEP' else 'walk')
                        if policy_switch.select(mode, env):
                            was_spin = spin
                            spin = False
                            if was_spin:
                                env.command[:2] = 0.
                            limits = policy_switch.command_limits
                            keyboard_speed = min(args.keyboard_speed if args.keyboard_speed is not None else
                                policy_switch.keyboard_commands[0], limits[0])
                            keyboard_yaw = min(args.keyboard_yaw if args.keyboard_yaw is not None else
                                policy_switch.keyboard_commands[1], limits[1])
                            env.command[:2] = env.command[:2].clip([-limits[0], -limits[1]], limits)
                            if handle is not None:
                                env.command[:2] = 0. if native_stop else held_axes * (keyboard_speed, keyboard_yaw)
                            if server is not None:
                                spin_toggle.value = False
                                spin_toggle.disabled = mode == 'step'
                                # 切回行走恢复完整滑条范围；台阶按自身范围限幅。
                                vx.min, vx.max = -limits[0], limits[0]
                                yaw.min, yaw.max = -limits[1], limits[1]
                                vx.value, yaw.value = map(float, env.command[:2])
                                height.max = policy_switch.height_max
                                height.value = float(env.command[2])
                                step_toggle.value = mode == 'step'
                            print(json.dumps(dict(event='policy_switch', **policy_switch.transitions[-1])), flush=True)
                    elif steps and key in {p['key'] for p in platforms}:
                        spin = False
                        if spin_toggle is not None: spin_toggle.value = False
                        selected_platform = next(p for p in platforms if p['key'] == key)
                        xy = selected_platform['spawn']
                        spawn = (*xy, ground_height(model, *xy, regional=True))
                        env.command[:2] = 0.
                        if server is not None: vx.value = yaw.value = 0.
                        env.reset(spawn, yaw=0.)
                        policy_switch.reset_parking()
                    elif regional and key in '123456':
                        spin = False
                        spots = [(-4., 4.), (.25, 4.), (8.25, 4.), (.25, 12.), (8.25, 12.), (4., -4.)]
                        xy = spots[int(key)-1]
                        spawn = (*xy, ground_height(model, *xy, regional=True))
                        env.command[:2] = 0.
                        env.reset(spawn, yaw=0.)
                    elif generated and key == 'V':
                        overview = not overview
                        with handle.lock():
                            handle.cam.distance = (47. if steps else 37.) if overview else 3.5
                            handle.cam.elevation = -65. if overview else -25.
                            handle.cam.azimuth = 90. if overview else 150.
                active_height_max = policy_switch.height_max if switching else height_max
                env.command[:] = env.command.clip([-limits[0], -limits[1], .22], [*limits, active_height_max])
                if not paused:
                    if arena_multi and policy_switch.gate is not None:
                        spin = False
                        env.command[:2] = env.command[:2].clip([-2.1, -2.], [2.1, 2.])
                    if switching:
                        policy_switch.apply_step_command(env.command)
                    # 小陀螺是持续目标模式，优先于方向键；X/复位显式退出。
                    if spin:
                        env.command[:2] = (0., 10.)
                        spin_steps += 1
                    if (regional or switching and policy_switch.mode == 'walk') and abs(env.command[1]) > 2.:
                        env.command[0] = 0.
                    # 指令变化同拍写入当前帧和最近一格历史，与GPU回放一致。
                    if arena_multi:
                        obs, history = policy_switch.inputs(env)
                    else:
                        obs, history = env.observation(), env.history
                        history[-19:-16] = obs[6:9]
                    previous_role = policy_switch.active_role if switching else None
                    action, _ = policy.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(history[None]))
                    if switching and policy_switch.active_role != previous_role:
                        print(json.dumps(dict(event='actor', time_s=float(env.data.time),
                            role=policy_switch.active_role, command=env.command.tolist())), flush=True)
                    jumping = arena_multi and policy_switch.gate is not None
                    if arena_multi:
                        samples.clear()
                    if allow_manual_recovery:
                        last_time = float(env.data.time)
                        last_qpos = env.data.qpos.copy()
                        last_qvel = env.data.qvel.copy()
                    try:
                        env.step(action[0].numpy(), direct=jumping,
                                 sample=sample_contacts if arena_multi and (jumping or args.output) else None)
                    except FloatingPointError:
                        if not allow_manual_recovery:
                            raise
                        warnings = {mujoco.mjtWarning(i).name: int(env.data.warning.lastinfo[i])
                                    for i, count in enumerate(env.data.warning.number) if count}
                        print(json.dumps(dict(event='numerical_stop', time_s=last_time, warnings=warnings,
                            role=policy_switch.active_role if switching else None, command=env.command.tolist(),
                            qpos=last_qpos.tolist(), qvel=last_qvel.tolist(), action=action[0].tolist())), flush=True)
                        # MuJoCo 可能已自动清空状态。用上一帧的有效位置停住画面，等 R 再重新起步。
                        mujoco.mj_resetData(model, env.data)
                        env.data.qpos[:] = last_qpos
                        env.data.time = last_time
                        numerical_stop = paused = True
                        continue
                    if arena_multi:
                        policy_switch.after_step(env, samples)
                        if jumping and policy_switch.gate is None:
                            print(json.dumps(dict(event='jump_handoff', **policy_switch.transitions[-1])), flush=True)
                        if args.output:
                            contact_trace.extend(samples)
                        if server is not None:
                            busy = policy_switch.gate is not None
                            step_toggle.disabled = busy
                            spin_toggle.disabled = busy or policy_switch.mode == 'step'
                            jump_button.disabled = spin_toggle.disabled
                    total_steps += 1
                    if generated:
                        for contact in env.data.contact:
                            if model.geom_bodyid[contact.geom1] == model.body('terrain').id or model.geom_bodyid[contact.geom2] == model.body('terrain').id:
                                contacts += 1
                                contact_error = max(contact_error, abs(float(contact.friction[0]) - .8))
                    if args.output is not None:
                        rows.append([total_steps * env.policy_dt, *env.command, *env.data.qpos[:3], *env.data.qvel[:6]])
                        poses.append(env.data.qpos.copy())
                        if switching:
                            policy_roles.append(policy_switch.active_role)
                # 按真实时间刷新，避免物理变慢后画面也只剩十几帧。
                if time.monotonic() >= next_draw:
                    if handle is not None:
                        with handle.lock():
                            handle.cam.lookat[:] = ((-5., 0., 0.) if steps else (0., 0., 0.)) if overview else env.data.qpos[:3]
                        handle.sync(state_only=True)
                        if numerical_stop:
                            handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                'Simulation stopped', 'Numerical warning. Press R to reset.'))
                        elif regional:
                            handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                f'Mode: {"SPIN" if spin else "DRIVE"}{" / PAUSED" if paused else ""}\n'
                                f'Target vx: {env.command[0]:+.1f} m/s\nTarget yaw: {env.command[1]:+.1f} rad/s\n'
                                f'Actual yaw: {env.data.qvel[5]:+.2f} rad/s',
                                f'G spin ON/OFF | X stop\nArrows: +/-{keyboard_speed:g}m/s, +/-{keyboard_yaw:g}rad/s\n'
                                '1 flat | 2-5 slopes | 6 rough\nR reset | P pause | V overview'))
                        elif arena_multi:
                            role = policy_switch.active_role
                            handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                f'Actor: {role.upper()} {policy_switch.metadata[role]["iterations"]} | '
                                f'{"SPIN" if spin else "DRIVE"}\nvx/yaw: {env.command[0]:+.1f}/{env.command[1]:+.1f}',
                                'T walk/step | Space jump | G spin\nArrows drive | X stop | R reset | P pause'))
                        elif steps:
                            step_arrows = (f'Up: {policy_switch.step_keyboard_speed:g}m/s | Release: park'
                                           if policy_switch.mode == 'step' and policy_switch.parking_enabled
                                           else f'Arrows: +/-{keyboard_speed:g}m/s, +/-{keyboard_yaw:g}rad/s')
                            handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                f'Mode: {policy_switch.mode.upper()} / Actor: {policy_switch.active_role.upper()} '
                                f'{policy_switch.metadata[policy_switch.active_role]["iterations"]}\n'
                                f'Platform: {selected_platform["height"] * 1000:g} mm\n'
                                f'Target vx/yaw: {env.command[0]:+.1f}/{env.command[1]:+.1f} | body: {env.command[2]:.2f} m\n'
                                f'{"SPIN" if spin else "DRIVE"} | Actual yaw: {env.data.qvel[5]:+.2f} rad/s',
                                f'{platform_keys}\n{step_arrows}\n'
                                'T walk/step | G spin (WALK) | X stop\nE/Q body | R reset | P pause | V overview'))
                    if scene is not None:
                        scene.update_from_mjdata(env.data)
                        if not switching:
                            status.content = f'模型 {metadata["iterations"]} · {"小陀螺" if spin else "普通行驶"} · 时间 {env.data.time:.2f}s · 指令 {env.command[0]:.1f}m/s，{env.command[1]:.1f}rad/s，{env.command[2]:.2f}m · 实际转速 {env.data.qvel[5]:.2f}rad/s\n\n原生MuJoCo场地体验；不作正式行为验收。'
                        if arena_multi:
                            role = policy_switch.active_role
                            status.content = (f'当前模型 {role.upper()} {policy_switch.metadata[role]["iterations"]} · '
                                f'{"小陀螺" if spin else "行驶"} · 速度/转向指令 {env.command[0]:+.1f}/{env.command[1]:+.1f}')
                        if steps:
                            status.content = (f'当前模型：{"上台阶" if policy_switch.mode == "step" else "行走"} '
                                f'{policy_switch.active_role.upper()} {policy_switch.metadata[policy_switch.active_role]["iterations"]}轮 · '
                                f'台阶 {selected_platform["height"] * 1000:g}mm · '
                                f'机身指令 {env.command[2]:.2f}m · 切模 {len(policy_switch.transitions)}次\n\n'
                                f'{"小陀螺" if spin else "普通行驶"} · 速度/转向指令 '
                                f'{env.command[0]:+.1f}m/s / {env.command[1]:+.1f}rad/s')
                    if handle is not None or scene is not None:
                        draw_count += 1
                    next_draw = time.monotonic() + 1. / (60. if handle is not None else 30.)
                if args.viewer != 'none':
                    period = .05 if paused else env.policy_dt
                    time.sleep(max(0., period - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        if handle is not None:
            handle.close()
            # 被动查看器在独立线程释放GL资源，等其结束后再释放模型。
            time.sleep(.25)
        if server is not None: server.stop()
        if args.output is not None:
            np.savez_compressed(args.output / 'trace.npz', state=np.asarray(rows), qpos=np.asarray(poses))
            if arena_multi:
                np.save(args.output / 'contact-substeps.npy', np.asarray(contact_trace))
            if switching:
                np.save(args.output / 'policy-roles.npy', np.asarray(policy_roles))
            report = dict(scope='interactive_native_playground_not_formal_acceptance', arena_revision=REVISION,
                arena=None if generated else str(args.arena.resolve()),
                arena_sha256=None if generated else hashlib.sha256(args.arena.read_bytes()).hexdigest(),
                terrain=terrain_record,
                friction=.8 if generated else None, ground_contacts=contacts,
                ground_contact_friction_max_error=contact_error,
                training_disturbances=saved['disturbances'], playback_disturbances=metadata['disturbances'],
                checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                iterations=metadata['iterations'], control=metadata['control'], timing=metadata['control_timing'],
                action_delay_steps=int(env.delays.action_steps[0]), sensor_delay_steps=int(env.delays.sensor_steps[0]),
                keyboard_speed=keyboard_speed, keyboard_yaw=keyboard_yaw,
                spin_initial=args.spin, spin_final=spin, spin_policy_steps=spin_steps,
                physics_engine='native_mujoco_cpu', mujoco=mujoco.__version__, ngeom=model.ngeom,
                spawn=spawn, spawn_yaw_deg=args.spawn_yaw_deg, command_sequence=schedule,
                task=metadata.get('task', 'fudan_ppo_walk'),
                policy_steps=total_steps, simulated_seconds=total_steps * env.policy_dt,
                wall_seconds=time.monotonic() - start, viewer_state_updates=draw_count,
                optimizer_updates=0, error=failure, formal_acceptance=False)
            if not generated:
                report['compiled_scene_key'] = cache.stem
            if without_rune:
                report['arena_variant'] = 'clear_centre_flat_fenced'
            if arena_multi:
                report['scheduled_keys'] = scheduled_keys
                report['contact_columns'] = ['left_vertical_N', 'right_vertical_N', 'max_other_force_N', 'base_vz_m_s']
            if switching:
                role = policy_switch.active_role
                current = policy_switch.checkpoints[role]
                report.update(checkpoint=current['checkpoint'], checkpoint_sha256=current['sha256'],
                    iterations=current['iterations'], task=policy_switch.metadata[role].get('task', 'fudan_ppo_walk'),
                    policy_switch=policy_switch.record(),
                    training_disturbances=policy_switch.training_disturbances[role],
                    playback_disturbances=policy_switch.metadata[role]['disturbances'])
            (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
