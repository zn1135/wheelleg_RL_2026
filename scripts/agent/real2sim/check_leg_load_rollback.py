"""Check the original plant on actual GPU training and native playback paths."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fitted_leg_load import PARAMETERS, parameter_record
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force
from ct_mjlab.envs.chuanliantui.huananhu_env_cfg import make_huananhu_env_cfg
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checks, evidence = {}, {}

    def verify(name, value):
        checks[name] = bool(value)
        assert checks[name], name

    raw = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    # 按真实名称查索引，避免训练装配后的关节编号变化造成误判。
    names = ('lf0', 'lf00', 'rf0', 'rf00') + tuple(raw.joint(int(i)).name for i in range(raw.njnt)
        if raw.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE)
    raw_ids = [raw.joint(name).dofadr[0] for name in names]
    nominal = {field: np.array(getattr(raw, field)[raw_ids])
               for field in ('dof_frictionloss', 'dof_damping')}
    verify('common_training_default_off', not make_huananhu_env_cfg(play=False).actions['pd'].fitted_leg_load)
    try:
        for randomized in (False, True):
            label = 'training' if randomized else 'nominal_gpu'
            wrapper = FudanEnv(20, seed=30001, randomized=randomized)
            env, action = wrapper.env, wrapper.action
            try:
                verify(label + '_no_fitted_load', action.leg_load is None)
                ids = [env.sim.mj_model.joint('robot/' + name).dofadr[0] for name in names]
                for field, expected in nominal.items():
                    verify(label + '_' + field, torch.allclose(getattr(env.sim.model, field)[..., ids],
                        torch.as_tensor(expected, device=wrapper.device, dtype=torch.float32)))
                calls, spring_errors = [], []
                original_step = env.sim.step

                def observed_step(*a, **kw):
                    calls.append(1)
                    verify(label + '_no_extra_applied_load', (env.sim.data.qfrc_applied[:, ids] == 0).all())
                    expected = gas_spring_force(env.sim.data.qpos[:, action._spring_q_adr],
                        env.sim.data.qvel[:, action._spring_v_adr], action._spring_force_bias)
                    spring_errors.append(float((env.sim.data.ctrl[:, action._spring_ids] - expected).abs().max()))
                    return original_step(*a, **kw)

                env.sim.step = observed_step
                wrapper.reset()
                for _ in range(30):
                    wrapper.step(torch.zeros(20, 6, device=wrapper.device), auto_reset=False, resample=False)
                verify(label + '_original_spring_each_step', len(calls) >= 150 and max(spring_errors) < .0001)
                verify(label + '_finite', torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all())
                evidence[label] = dict(physics_steps=len(calls), spring_max_error_n=max(spring_errors),
                    profile=parameter_record(action.leg_load is not None),
                    actual={field: getattr(env.sim.model, field)[..., ids].cpu().tolist() for field in nominal})
            finally:
                env.close()
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'sim2sim'))
        from mj_sim2sim_huananhu_multimode import Playback
        from mj_sim2sim_chuanliantui_stand_v2 import StandV2Playback
        verify('stand_playback_default_off', inspect.signature(StandV2Playback).parameters['fitted_leg_load'].default is False)
        playback = Playback('normal')
        verify('native_playback_no_fitted_load', playback.leg_load is None)
        native_ids = [playback.model.joint(name).dofadr[0] for name in names]
        for field, expected in nominal.items():
            verify('native_' + field, np.array_equal(getattr(playback.model, field)[native_ids], expected))
        evidence['preserved_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (MODEL_XML, PARAMETERS)}
    finally:
        result = dict(passed=len(checks) >= 17 and all(checks.values()), checks=checks, evidence=evidence,
            physics_backend='mujoco_warp_gpu', policy_training_updates=0, hardware_commands=0,
            scope='Actual20-world nominal/training GPU forces and native construction; not full motion acceptance')
        (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
