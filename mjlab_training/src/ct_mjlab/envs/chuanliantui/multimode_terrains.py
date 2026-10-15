"""Simple sloped box patch, avoiding dense HFIELD/MESH contact pairs."""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class BoxSlopeTerrainCfg(SubTerrainCfg):
    gradient_range: tuple[float, float] = (.087, .176)

    def function(self, difficulty, spec, rng):
        del rng
        gradient = self.gradient_range[0] + difficulty * (
            self.gradient_range[1] - self.gradient_range[0])
        angle = math.atan(gradient)
        half = angle / 2.
        geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2, self.size[1] / 2, .08),
            pos=(self.size[0] / 2, self.size[1] / 2, -.08),
            quat=(math.cos(half), 0., math.sin(half), 0.),
        )
        return TerrainOutput(
            origin=np.array((self.size[0] / 2, self.size[1] / 2, 0.)),
            geometries=[TerrainGeometry(geom=geom)])
