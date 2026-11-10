"""Check whether a slider end reaction can support the measured initial pose; zero integration."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim.fit_leg_response_gpu import set_extension_stop_xml


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    reference = Path('artifacts/real2sim/h7-response-align-20261104-01/final-elastic-replay')
    tree = ET.parse(reference/'fixed-base-model.xml').getroot()
    set_extension_stop_xml(tree, [dict(spring_extension_stop_m=[.007]*2, spring_extension_margin_m=[.001]*2)])
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    mq = [model.joint(n).dofadr[0] for n in ('lf0', 'lf00', 'rf0', 'rf00')]
    slider = [model.joint(s+'_spring2_joint').id for s in ('left', 'right')]
    sq = model.jnt_qposadr[slider]; sv = model.jnt_dofadr[slider]
    spring = [model.actuator(s+'_gas_spring_motor').id for s in ('left', 'right')]
    motors = [model.actuator(n+'_motor').id for n in ('lf0', 'lf00', 'rf0', 'rf00')]
    model.dof_armature[mq] = .015795; model.dof_frictionloss[mq] = [2.2, 2.8, 2.2, 2.8]
    model.dof_frictionloss[sv] = 30.; model.dof_damping[mq] = .3
    model.dof_solref[mq] = [.002, 1.]; model.dof_solimp[mq] = [.999, .9999, .001, .5, 2.]
    model.dof_solref[sv] = [.002, 1.]; model.dof_solimp[sv] = [.999, .9999, .001, .5, 2.]
    mujoco.mj_setConst(model, mujoco.MjData(model))
    with np.load(reference/'replay.npz') as recorded:
        initial = recorded['qpos'][0, 0].copy(); control = recorded['submitted_model_torque'][0, 0, :4].copy()
    rows = []
    for endpoint in np.linspace(.0068, .0076, 41):
        model.jnt_range[slider, 1] = endpoint
        data = mujoco.MjData(model); data.qpos[:] = initial; data.qvel[:] = 0.; data.ctrl[motors] = control
        compression = np.maximum(0., .010-initial[sq]); data.ctrl[spring] = 150.+750.*(compression-.005)
        mujoco.mj_forward(model, data)
        reactions = np.zeros(2)
        for i in range(data.nefc):
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
                indices = np.flatnonzero(np.asarray(slider) == data.efc_id[i])
                if len(indices): reactions[indices[0]] += data.efc_force[i]
        rows.append(dict(endpoint_m=float(endpoint), margin_m=.001, motor_acceleration_rad_s2=data.qacc[mq].tolist(),
                         slider_end_reaction_n=reactions.tolist(), maximum_motor_acceleration_rad_s2=float(max(abs(data.qacc[mq])))))
    best = min(rows, key=lambda row: row['maximum_motor_acceleration_rad_s2'])
    report = dict(scope='Native forward force balance only; not GPU response acceptance or measured stroke calibration',
                  environment_steps=0, source_reference=str(reference), reference_sha256=hashlib.sha256((reference/'replay.npz').read_bytes()).hexdigest(),
                  initial_slider_position_m=initial[sq].tolist(), motor_friction_nm=[2.2, 2.8, 2.2, 2.8],
                  slider_friction_n=30., spring_reference_force_n=150., spring_stiffness_n_m=750.,
                  results=rows, best_static_balance=best,
                  limitations=['Endpoint is an unmeasured diagnostic hypothesis', 'Static equilibrium does not establish a continuous dynamic fit'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(best, indent=2))


if __name__ == '__main__': main()
