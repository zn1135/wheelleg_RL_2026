"""Compare GPU solvers/integration on the same captured bounded-force input history."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs.fudan_ppo_walk import fudan_ppo_walk as task


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with np.load(args.input) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    records = []
    original_cfg = task.make_env_cfg
    for solver, dt, substeps in (('newton', .002, 1), ('cg', .002, 1), ('newton', .001, 2), ('newton', .0005, 4)):
        def diagnostic_cfg(*a, **kw):
            cfg = original_cfg(*a, **kw)
            cfg.sim.mujoco.solver = solver
            cfg.sim.mujoco.timestep = dt
            return cfg
        with patch.object(task, 'make_env_cfg', diagnostic_cfg):
            wrapper = task.FudanEnv(1, seed=1, randomized=True)
        sim = wrapper.env.sim
        outputs, bad_at = [], None
        try:
            for key, value in saved.items():
                if key.startswith('model__'):
                    getattr(sim.model, key.removeprefix('model__')).copy_(torch.as_tensor(value, device=wrapper.device))
            # 保持原2ms步长的refsafe有效时间常数，避免缩步同时改硬度。
            for field in ('eq_solref', 'jnt_solref', 'dof_solref'):
                if hasattr(sim.model, field):
                    first = getattr(sim.model, field)[..., 0]
                    first.copy_(torch.where(first > 0, first.clamp(min=.004), first))
            sim.data.qpos.copy_(torch.as_tensor(saved['qpos_history'][0], device=wrapper.device))
            sim.data.qvel.copy_(torch.as_tensor(saved['qvel_history'][0], device=wrapper.device))
            sim.forward()
            for i, ctrl in enumerate(saved['ctrl_history']):
                sim.data.ctrl.copy_(torch.as_tensor(ctrl, device=wrapper.device))
                sim.data.xfrc_applied.copy_(torch.as_tensor(saved['xfrc_applied_history'][i], device=wrapper.device))
                for _ in range(substeps):
                    sim.step()
                outputs.append(sim.data.qvel.cpu().numpy().copy())
                if not torch.isfinite(sim.data.qpos).all() or not torch.isfinite(sim.data.qvel).all():
                    bad_at = i
                    break
            result = dict(solver=solver, integration_dt=dt, substeps_per_original2ms=substeps,
                input_intervals=len(saved['ctrl_history']), replayed_intervals=len(outputs), nonfinite_at=bad_at,
                max_abs_qvel=float(np.nanmax(abs(np.asarray(outputs)))) if outputs else None)
            np.savez_compressed(args.output/f'{solver}-{dt}.npz', qvel=np.asarray(outputs))
            records.append(result)
            print(json.dumps(result), flush=True)
        finally:
            wrapper.env.close()
    (args.output/'report.json').write_text(json.dumps(dict(variants=records,optimizer_updates=0,
        scope='Same captured input, one-world GPU replay; warm-start/constraint history not fully restored, not exact4096-world replay or behavior acceptance'),indent=2)+'\n')


if __name__ == '__main__':
    main()
