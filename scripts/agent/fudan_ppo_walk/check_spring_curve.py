"""Check the selected spring curve on real training/playback paths; no integration."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui import gas_spring as spring
from ct_mjlab.envs.chuanliantui.model import apply_gas_spring_forces, gas_spring_actuator_ids
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import make_env_cfg, PhysicalAction
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config, from_sim2sim_config
from .check_disturbances import control_fixture

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import mj_sim2sim_huananhu_multimode as native


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def verify(name, condition):
        assert bool(condition), name
        checks.append(name)

    def forbidden(*args, **kwargs):
        raise AssertionError('Integration, network forward and optimizer updates are forbidden')

    # 包含伸长/压缩速度和两次夹紧，不能仅凭静止弹性值判定阻尼保持。
    positions = np.array([.010, 0., -.080, 0., 0., .100, -.100, -.080, .010])
    velocities = np.array([0., 0., 0., .1, -.1, 0., 0., -.1, .1])
    expected = np.array([0., 150/9, 150., 150/9-5, 150/9+5, 0., 150., 150., 0.])
    with ExitStack() as stack:
        for name in ('mujoco.mj_step', 'mujoco.mj_step1', 'mujoco.mj_step2',
                     'mjlab.envs.ManagerBasedRlEnv.step', 'torch.nn.Linear.forward',
                     'torch.optim.Adam.step'):
            stack.enter_context(patch(name, side_effect=forbidden))
        verify('parameters_90mm_10mm_150N_damping50',
               spring.SPRING_REFERENCE_STROKE == .090 and spring.SPRING_INITIAL_COMPRESSION == .010
               and spring.SPRING_FORCE_MIN == 0. and spring.SPRING_FORCE_LIMIT == 150.
               and spring.SPRING_DAMPING == 50. and native.SPRING_DAMPING == 50.)
        verify('numpy_training_curve_and_damping', np.allclose(spring.gas_spring_force(positions, velocities), expected))
        verify('native_playback_curve_and_damping', np.allclose(native.gas_spring_force(positions, velocities), expected))
        for device in ('cpu', 'cuda:0'):
            force = spring.gas_spring_force(torch.tensor(positions, device=device),
                                           torch.tensor(velocities, device=device))
            verify(f'torch_{device}_matches_independent_values', np.allclose(force.cpu().numpy(), expected))
        for play in (False, True):
            cfg = make_env_cfg(play=play).actions['pd']
            verify(f'actual_config_play_{play}_uses_physical_action', cfg.class_type is PhysicalAction)
            verify(f'actual_config_play_{play}_has_zero_spring_bias', cfg.spring_force_bias_range == (0., 0.))
        for label, disturbances in (('training', from_task_config()), ('sim2sim', from_sim2sim_config())):
            env = control_fixture(disturbances)
            action = env.action
            action._spring_force_bias.zero_()  # 名义对照按实际任务配置关闭弹簧偏置。
            verify(f'{label}_actual_action_keeps_damping50', action.spring_damping.eq(50.).all())
            action.process_actions(torch.zeros(3, 6))
            for offset in range(0, len(positions), 3):
                q = torch.tensor(positions[offset:offset+3], dtype=torch.float32)[:, None].expand(-1, 2)
                v = torch.tensor(velocities[offset:offset+3], dtype=torch.float32)[:, None].expand(-1, 2)
                env.env.sim.data.qpos[:, action._spring_q_adr] = q
                env.env.sim.data.qvel[:, action._spring_v_adr] = v
                action.apply_actions()
                applied = env.env.sim.data.ctrl[:, action._spring_ids].cpu().numpy()
                verify(f'{label}_actual_actuator_write_cases_{offset}',
                       np.allclose(applied, expected[offset:offset+3, None], atol=2e-5))
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'))
        data = mujoco.MjData(model)
        actuators = gas_spring_actuator_ids(model)
        joints = model.actuator_trnid[actuators, 0]
        for index, (q, v, target) in enumerate(zip(positions, velocities, expected)):
            data.qpos[model.jnt_qposadr[joints]] = q
            data.qvel[model.jnt_dofadr[joints]] = v
            force = apply_gas_spring_forces(model, data)
            verify(f'native_shared_actuator_write_case_{index}',
                   np.allclose(force, target) and np.allclose(data.ctrl[actuators], target))
    sources = [Path(spring.__file__), Path(native.__file__),
               ROOT / 'mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_action.py',
               ROOT / 'mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk.py',
               ROOT / 'mjlab_training/src/ct_mjlab/envs/chuanliantui/model.py']
    report = dict(passed=True, count=len(checks), checks=checks, stroke_mm=90.,
                  zero_pose_compression_mm=10., elastic_force_range_n=[0., 150.], damping_ns_m=50.,
                  physics_steps=0, network_forwards=0, optimizer_updates=0,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  scope='Real action/actuator methods on synthetic states plus NumPy/Torch CPU/CUDA; not behavior acceptance')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
