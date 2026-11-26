"""少量盒体组成单级平台；各行是高度课程，各列是平地/上台/下台。"""
from dataclasses import dataclass
import mujoco
import numpy as np
from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.sensor import GridPatternCfg


@dataclass(kw_only=True)
class PlatformCfg(SubTerrainCfg):
    mode: str = 'up'
    heights: tuple = (0., .02, .05, .08, .12, .16, .18, .20, .21, .22)
    width: float = 1.148
    depth: float = 2.09
    side_obstacle_level: int = 7
    side_obstacle_height: float = .55
    approach_distance: tuple = (0., 1.2, .6)

    def function(self, difficulty, spec, rng):
        del rng
        level = round(difficulty * (len(self.heights) - 1))
        height = 0. if self.mode == 'flat' else self.heights[level]
        body = spec.body('terrain')
        geoms = []

        def box(size, pos):
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size, pos=pos, group=0, friction=(1., .005, .0001))
            geoms.append(TerrainGeometry(geom=geom))

        box((self.size[0]/2, self.size[1]/2, .1), (self.size[0]/2, self.size[1]/2, -.1))
        if height > 0:
            box((self.depth/2, self.width/2, height/2), (4.+self.depth/2, 2., height/2))
        if self.mode != 'flat' and level >= self.side_obstacle_level:
            for sign in (-1, 1):
                box((.35, .2, self.side_obstacle_height/2),
                    (4., 2.+sign*(self.width/2+.2), self.side_obstacle_height/2))
        spawn_x = {'flat': 5., 'up': 4.-self.approach_distance[1],
                   'down': 4.+self.approach_distance[2]}[self.mode]
        return TerrainOutput(origin=np.array((spawn_x, 2., height if self.mode == 'down' else 0.)),
                             geometries=geoms)


@dataclass
class ElevatedGrid(GridPatternCfg):
    """从机身/轮轴上方发射，防止台面高于轮轴时射线漏掉台面。"""
    def generate_rays(self, mj_model=None, device='cpu'):
        origins, directions = super().generate_rays(mj_model, device)
        origins[:, 2] += 1.
        return origins, directions
