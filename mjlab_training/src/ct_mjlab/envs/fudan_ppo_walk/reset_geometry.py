"""Solve legal closed poses at the unchanged physical PD centres; no time stepping."""
from functools import lru_cache
import numpy as np
import mujoco
from scipy.optimize import least_squares
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS, _SIDES, load_model, solve_standing_joint_positions, closed_chain_residual
RESET_CLEARANCE_M = 0.002

def mesh_vertices(model, data, geom_id):
    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    return model.mesh_vert[start:start + count] @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]

class ResetGeometry:

    def __init__(self):
        self.model = load_model()
        self.baseline = solve_standing_joint_positions(self.model)
        self.data = mujoco.MjData(self.model)
        self.joint_names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
        self.qaddr = {name: int(self.model.joint(name).qposadr[0]) for name in self.joint_names}
        self.wheels = {self.model.body(n).id for n in ('lfwheel', 'rfwheel')}
        self.collision = [i for i in range(self.model.ngeom) if self.model.geom_contype[i] or self.model.geom_conaffinity[i]]

    def solve(self, nominal):
        nominal = np.asarray(nominal, dtype=np.float64)
        model, data = (self.model, self.data)
        data.qpos[:] = model.qpos0
        data.qpos[:7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        for name, value in self.baseline.items():
            data.qpos[self.qaddr[name]] = value
        for name, value in zip(ACTUATED_JOINTS, nominal, strict=True):
            data.qpos[self.qaddr[name]] = value
        for side, hip, knee, passive, pairs, _ in _SIDES:
            names = (knee, *passive[1:])
            addresses = [self.qaddr[name] for name in names]
            lower = []
            upper = []
            for name in names:
                joint = model.joint(name)
                lo, hi = joint.range if joint.limited else (model.qpos0[self.qaddr[name]] - np.pi, model.qpos0[self.qaddr[name]] + np.pi)
                lower.append(lo)
                upper.append(hi)
            site_pairs = [(model.site(a).id, model.site(b).id) for a, b in pairs]

            def residual(values):
                data.qpos[addresses] = values
                mujoco.mj_kinematics(model, data)
                return np.concatenate([data.site_xpos[a] - data.site_xpos[b] for a, b in site_pairs])
            result = least_squares(
                residual,
                data.qpos[addresses].copy(),
                bounds=(lower, upper),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=1000
            )
            error = float(np.max(np.abs(residual(result.x))))
            if not result.success or error > 1e-08:
                raise ValueError(f'PD centre has no verified legal {side} reset: nominal={nominal.tolist()}, residual={error}, passive={result.x.tolist()}, bounds={list(zip(lower, upper))}')
        mujoco.mj_kinematics(model, data)
        for side, parent in (('left', 'lf0'), ('right', 'rf0')):
            delta = data.body(parent).xmat.reshape(3, 3).T @ (data.site(f'{side}_gas_spring_lower').xpos - data.site(f'{side}_gas_spring_upper').xpos)
            quat = model.body(f'{side}_spring1').quat
            angle = 2 * np.arctan2(quat[2], quat[0])
            length = model.body(f'{side}_spring2').pos[2]
            data.qpos[self.qaddr[f'{side}_spring1_joint']] = np.arctan2(delta[0], delta[2]) - angle
            data.qpos[self.qaddr[f'{side}_spring2_joint']] = np.hypot(delta[0], delta[2]) - length
        mujoco.mj_kinematics(model, data)
        for name in self.joint_names:
            joint = model.joint(name)
            q = data.qpos[self.qaddr[name]]
            if joint.limited and (not joint.range[0] - 1e-09 <= q <= joint.range[1] + 1e-09):
                raise ValueError(f'PD centre reset exceeds {name}: q={q}, range={joint.range.tolist()}')
        residual = closed_chain_residual(model, data)
        if residual > 1e-08:
            raise ValueError(f'PD centre reset does not close: {residual}')
        wheel_bottom = min((float(mesh_vertices(model, data, i)[:, 2].min()) for i in self.collision if int(model.geom_bodyid[i]) in self.wheels))
        all_bottom = min((float(mesh_vertices(model, data, i)[:, 2].min()) for i in self.collision))
        if all_bottom < wheel_bottom - 1e-06:
            raise ValueError(f'PD centre cannot reset wheel-first: body bottom={all_bottom}, wheel bottom={wheel_bottom}')
        height = -wheel_bottom + RESET_CLEARANCE_M
        positions = {name: float(np.float32(data.qpos[self.qaddr[name]])) for name in self.joint_names}
        height = float(np.float32(height))
        data.qpos[:7] = (0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0)
        for name, value in positions.items():
            data.qpos[self.qaddr[name]] = value
        mujoco.mj_kinematics(model, data)
        written_residual = closed_chain_residual(model, data)
        written_clearance = min((float(mesh_vertices(model, data, i)[:, 2].min()) for i in self.collision))
        for name, value in positions.items():
            joint = model.joint(name)
            if joint.limited and (not joint.range[0] - 1e-08 <= value <= joint.range[1] + 1e-08):
                raise ValueError(f'Float32 reset exceeds {name}: {value}')
        if written_residual > 1e-06 or written_clearance < 0.0:
            raise ValueError(f'Float32 reset is invalid: residual={written_residual}, clearance={written_clearance}')
        return dict(
            height=height,
            joint_positions=positions,
            closed_chain_residual_m=residual,
            minimum_ground_clearance_m=written_clearance,
            written_float32_residual_m=written_residual,
            nominal_motor_positions=nominal.tolist()
        )

@lru_cache(maxsize=1)
def nominal_reset_pose():
    solver = ResetGeometry()
    return solver.solve([solver.baseline[name] for name in ACTUATED_JOINTS])
