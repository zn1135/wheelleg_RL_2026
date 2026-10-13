"""Robot adapter: retain this repository's CAD, refs and solved pose cache."""
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from ..asset import robot_spec, POSES
from ..config import ASSET

MODEL_XML = ASSET
ACTUATED_JOINTS = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
# Policy outputs and observation slots use four legs followed by two wheels.
# Keep the physical actuator layout independent of this public interface.
POLICY_JOINTS = ('lf0', 'lf00', 'rf0', 'rf00', 'lfwheel', 'rfwheel')
GAS_SPRING_JOINTS = ('left_spring2_joint', 'right_spring2_joint')

def gas_spring_actuator_ids(model):
    import mujoco
    prefix = 'robot/' if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot/base_link') >= 0 else ''
    return [model.actuator(prefix + side + '_gas_spring_motor').id for side in ('left','right')]

def standing_motor_positions():
    return tuple(POSES['nominal'][name] for name in ('lf0','lf00','rf0','rf00'))

def create_robot_cfg(*, base_height=.3376):
    return EntityCfg(spec_fn=robot_spec, sort_actuators=False,
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=ACTUATED_JOINTS),)),
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,base_height),
            joint_pos={f'^{name}$': value for name,value in POSES['nominal'].items()},
            joint_vel={'.*':0.}))
