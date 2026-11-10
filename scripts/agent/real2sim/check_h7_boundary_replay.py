"""Verify that the new versioned boundary controller preserves old GPU replays."""
import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from scripts.agent.real2sim.replay_leg_test_gpu import training_spring


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    old=json.loads((args.before/'report.json').read_text())
    new=json.loads((args.after/'report.json').read_text())
    assert new['leg_boundary_mode']=='legacy_lockout'
    assert new['physics_backend']=='mujoco_warp_gpu' and new['new_environment_steps']==16000
    for key in ('stream_sha256','source_model_sha256','config_sha256','response_parameters_sha256',
                'spring_model','training_spring_source_sha256','window_start_board_s','duration_s'):
        assert old[key]==new[key],key
    assert (args.before/'fixed-base-model.xml').read_bytes()==(args.after/'fixed-base-model.xml').read_bytes()
    model=mujoco.MjModel.from_xml_path(str(args.after/'fixed-base-model.xml'))
    sq=[model.joint(side+'_spring2_joint').qposadr[0] for side in ('left','right')]
    sv=[model.joint(side+'_spring2_joint').dofadr[0] for side in ('left','right')]
    with np.load(args.before/'replay.npz') as before,np.load(args.after/'replay.npz') as after:
        for key in ('time_s','real_geometry','real_motor_state','initial_qpos','initial_qvel'):
            assert np.array_equal(before[key],after[key]),key
        # 实录力矩才是输入；另一分支力矩由各自的GPU反馈计算，不能要求逐位相等。
        assert np.array_equal(before['submitted_model_torque'][:,0],after['submitted_model_torque'][:,0])
        feedback_difference=float(np.max(np.abs(before['submitted_model_torque'][:,1]-after['submitted_model_torque'][:,1])))
        differences={key:float(np.max(np.abs(before[key]-after[key]))) for key in ('qpos','qvel','elastic_resistance_nm','spring_force_n')}
        force_errors=[]
        for data in (before,after):
            expected=training_spring.gas_spring_force(data['qpos'][:-1,:,sq].astype(np.float32),data['qvel'][:-1,:,sv].astype(np.float32))
            force_errors.append(float(np.max(np.abs(expected-data['spring_force_n']))))
    assert differences['qpos']<1e-4 and differences['qvel']<1e-3
    assert differences['elastic_resistance_nm']<1e-3 and max(force_errors)<2e-5
    assert feedback_difference<.02
    result=dict(passed=True,scope='old-record GPU compatibility only; no new-firmware hardware or powered acceptance',
                max_differences=differences,old_controller_retained=True,recorded_inputs_identical=True,
                max_feedback_request_difference_nm=feedback_difference,spring_law_error_n=force_errors,
                compared_gpu_environment_steps=new['total_environment_steps'],additional_steps_in_this_check=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:json.dump(result,file,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
