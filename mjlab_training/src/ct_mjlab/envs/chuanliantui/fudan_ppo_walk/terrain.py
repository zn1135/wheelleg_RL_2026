"""连续分区：左半平地，右上四座金字塔坡，右下小崎岖。"""
from dataclasses import dataclass
import math
import mujoco
import numpy as np
import torch
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.utils.spec_config import GeomCfg

VERSION = 'fudan-continuous-pyramid-rough-regions-v2'
PLANE = dict(version='fudan-plane-v1', mode='plane')
LEGACY_CRITIC = dict(version='fudan-critic-repeated-height-v1')


def critic_record(cfg):
    if not cfg.observations.critic_height_scan:
        return dict(LEGACY_CRITIC)
    if cfg.terrain.mode != 'regions':
        raise ValueError('Walk critic height scan requires regional terrain')
    return dict(version='fudan-critic-yaw-height-grid-v1', slots=[46, 123],
        grid_size_m=[1., .6], resolution_m=.1, num_rays=77,
        alignment='yaw', order='xy_meshgrid_x_fastest', origin_above_body_m=1.,
        max_distance_m=3., geom_groups=[0],
        value='clamp(root_z-ground_z-height_center,-1,1)*height_scale',
        height_center=cfg.normalization.height_center, height_scale=cfg.normalization.height_scale,
        actor_terrain_input=False, missing_ray='terminate_zero_reward_no_bootstrap')


def record(cfg):
    terrain = cfg.terrain
    if terrain.mode == 'plane':
        return dict(PLANE)
    if terrain.mode != 'regions':
        raise ValueError('Unsupported walk terrain mode')
    if not (terrain.size >= 8 and 0 < terrain.resolution <= .25
            and terrain.resolution <= terrain.rough_spacing
            and 0 < terrain.pyramid_platform < terrain.size/4
            and 0 < terrain.slope_degrees[0] <= terrain.slope_degrees[1] < 30
            and 0 <= terrain.rough_amplitude <= .05):
        raise ValueError('Invalid regional terrain dimensions')
    return dict(version=VERSION, **vars(terrain), area_ratios=[.5, .25, .25],
        spawn_ratios=[.5, .25, .25], spawn='fixed_region_centres_on_flat_pads',
        height_reference='vertical_clearance_above_current_ground', actor_terrain_input=False,
        collision_geometry='flat_boxes+convex_pyramids+rough_heightfield')


def height_map(terrain):
    size = terrain.size
    n = round(size/terrain.resolution)+1
    axis = np.linspace(0., size, n)
    xx, yy = np.meshgrid(axis, axis)
    height = np.zeros_like(xx)
    # 金字塔由四个平面构成；用距方形边缘的最小距离，避免生成圆锥或双线性弧坡。
    tile = size/4
    slopes = np.linspace(*terrain.slope_degrees, 4)
    centres = []
    for k, (ix, iy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        left, bottom = size/2+ix*tile, size/2+iy*tile
        distance = np.minimum.reduce((xx-left, left+tile-xx, yy-bottom, bottom+tile-yy))
        rise = math.tan(math.radians(slopes[k])) * np.clip(
            distance, 0., (tile-terrain.pyramid_platform)/2)
        height = np.maximum(height, rise)
        centres.append((left+tile/2, bottom+tile/2, float(rise.max())))
    rng = np.random.default_rng(terrain.seed)
    coarse_x = np.linspace(size/2, size, round(size/2/terrain.rough_spacing)+1)
    coarse_y = np.linspace(0., size/2, round(size/2/terrain.rough_spacing)+1)
    coarse = rng.uniform(-terrain.rough_amplitude, terrain.rough_amplitude,
                         (len(coarse_y), len(coarse_x)))
    across = np.array([np.interp(axis, coarse_x, row) for row in coarse])
    noise = np.array([np.interp(axis, coarse_y, col) for col in across.T]).T
    taper = np.clip(np.minimum.reduce((xx-size/2, size-xx, yy, size/2-yy)), 0., 1.)
    # 复位点留1m平坦落脚区，外侧0.5m渐变回崎岖路面。
    rough_centres = []
    for ix, iy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        cx, cy = size/2+(ix+.5)*tile, (iy+.5)*tile
        distance = np.maximum(np.abs(xx-cx), np.abs(yy-cy))
        taper *= np.clip((distance-.5)/.5, 0., 1.)
        rough_centres.append((cx, cy, 0.))
    height += noise*taper
    flat_centres = [(size/8, size/4, 0.), (3*size/8, size/4, 0.),
                    (size/8, 3*size/4, 0.), (3*size/8, 3*size/4, 0.)]
    spawns = np.array(flat_centres+centres+rough_centres)
    spawns[:, :2] -= size/2
    return axis-size/2, height, spawns


@dataclass(kw_only=True)
class RegionalTerrainCfg(SubTerrainCfg):
    terrain: object

    def function(self, difficulty, spec, rng):
        del difficulty, rng
        axis, height, centres = height_map(self.terrain)
        size = self.terrain.size
        body = spec.body('terrain')
        geometries = []
        for name, halfsize, pos in (
            ('walk_flat', (size/4, size/2, .05), (size/4, size/2, -.05)),
            ('walk_pyramid_base', (size/4, size/4, .05), (3*size/4, 3*size/4, -.05)),
        ):
            geom = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=halfsize, pos=pos, group=0, friction=(1., .005, .0001))
            geometries.append(TerrainGeometry(geom=geom))
        for i, (cx, cy, top) in enumerate(centres[4:8]):
            vertices = [(sx*size/8, sy*size/8, z) for z in (-.1, 0.) for sx, sy in
                        ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            vertices += [(sx*self.terrain.pyramid_platform/2, sy*self.terrain.pyramid_platform/2, top)
                         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            name = f'walk_pyramid_{i}'
            spec.add_mesh(name=name, uservert=np.asarray(vertices).ravel().tolist())
            geom = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                pos=(cx+size/2, cy+size/2, 0.), group=0, friction=(1., .005, .0001), rgba=(.6, .65, .5, 1.))
            geometries.append(TerrainGeometry(geom=geom))
        # 只有小崎岖使用高度场；20cm原始噪声节点直接参与碰撞，不扩大起伏范围。
        mid = len(axis)//2
        rough = height[:mid+1, mid:]
        low, high = float(rough.min()), float(rough.max())
        span = max(high-low, 1e-6)
        field = spec.add_hfield(name='walk_rough', size=[size/4, size/4, span, .05],
            nrow=len(rough), ncol=len(rough), userdata=((rough-low)/span).ravel().tolist())
        geom = body.add_geom(name='walk_rough', type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name, pos=(3*size/4, size/4, low), group=0,
            friction=(1., .005, .0001), rgba=(.55, .68, .55, 1.))
        geometries.append(TerrainGeometry(geom=geom, hfield=field))
        # 仅作外场底面，正常回合在距场地边缘1m处结束，不允许落到无地面区域。
        floor = body.add_geom(name='walk_outer_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0., 0., .1), pos=(0., 0., -.1), group=0, friction=(1., .005, .0001))
        geometries.append(TerrainGeometry(geom=floor))
        return TerrainOutput(origin=np.array((size/4, size/2, 0.)), geometries=geometries)


@dataclass
class GroundRay(GridPatternCfg):
    def generate_rays(self, mj_model=None, device='cpu'):
        origins, directions = super().generate_rays(mj_model, device)
        origins[:, 2] += 1.
        return origins, directions


def reset_regions(env, env_ids):
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import reset_pose
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, '_region_spawns'):
        _, _, centres = height_map(env.cfg._fudan_task_cfg.terrain)
        index = torch.arange(env.num_envs, device=env.device)
        kind = torch.where(index*4 < env.num_envs*2, 0,
                           torch.where(index*4 < env.num_envs*3, 1, 2))
        env._region_kind = kind
        env._region_spawns = torch.as_tensor(centres, device=env.device, dtype=torch.float32)[kind*4+index%4]
    env.scene.terrain.env_origins[env_ids] = env._region_spawns[env_ids]
    reset_pose(env, env_ids)


@requires_model_fields('body_mass', 'body_inertia', 'body_ipos', 'geom_friction',
                       'geom_solref', recompute=RecomputeLevel.set_const)
def randomize_regions(env, env_ids):
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import randomize
    model = env.sim.mj_model
    body = model.body('terrain').id
    env._terrain_friction_ids = torch.tensor([i for i in range(model.ngeom)
        if model.geom_bodyid[i] == body], device=env.device)
    randomize(env, env_ids)


def configure(physics, task):
    record(task)
    critic_record(task)
    if task.terrain.mode == 'plane':
        return
    terrain = physics.scene.terrain
    terrain.terrain_type = 'generator'
    terrain.materials, terrain.textures = (), ()
    terrain.geoms = (GeomCfg(geom_names_expr=('.*',), friction=(1., .005, .0001)),)
    terrain.terrain_generator = TerrainGeneratorCfg(seed=task.terrain.seed, curriculum=True,
        size=(task.terrain.size, task.terrain.size), num_rows=1, num_cols=1, color_scheme='none',
        sub_terrains={'regions': RegionalTerrainCfg(terrain=task.terrain)})
    physics.events['reset_pose'] = EventTermCfg(func=reset_regions, mode='reset')
    if 'fudan_randomization' in physics.events:
        physics.events['fudan_randomization'] = EventTermCfg(func=randomize_regions, mode='startup')
    sensor = RayCastSensorCfg(name='walk_ground', frame=ObjRef(type='body', name='base_link', entity='robot'),
        pattern=GroundRay(size=(0., 0.)), ray_alignment='world', max_distance=3., include_geom_groups=(0,))
    physics.scene.sensors = (*physics.scene.sensors, sensor)
    if task.observations.critic_height_scan:
        scan = RayCastSensorCfg(name='walk_scan', frame=ObjRef(type='body', name='base_link', entity='robot'),
            pattern=GroundRay(size=(1., .6), resolution=.1), ray_alignment='yaw',
            max_distance=3., include_geom_groups=(0,))
        physics.scene.sensors = (*physics.scene.sensors, scan)


def scan_state(env):
    ray = env.env.scene['walk_scan'].data
    heights = ray.hit_pos_w[..., 2]
    invalid = ((ray.distances < 0) | ~torch.isfinite(heights)).any(-1)
    # 仅为终止拍保留有限输入；缺测回合由terrain_invalid清奖励、禁bootstrap。
    heights = torch.where(invalid[:, None], env.env.scene.env_origins[:, 2, None], heights)
    return heights, invalid


def ground_state(env):
    ray = env.env.scene['walk_ground'].data
    ground = ray.hit_pos_w[:, 0, 2]
    position = env.robot.data.root_link_pos_w
    outside = (position[:, :2].abs() > env.cfg.terrain.size/2-1.).any(-1)
    invalid = (ray.distances[:, 0] < 0) | ~torch.isfinite(ground) | outside
    # 缺测只给终止拍保留有限张量，invalid明确结束该回合并清奖励。
    safe_ground = torch.where(invalid, env.env.scene.env_origins[:, 2], ground)
    return position[:, 2]-safe_ground, invalid
