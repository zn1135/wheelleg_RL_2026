"""2027 robot's independent walk parameters."""
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, FudanWalkCfgPPO
from .model import standing_motor_positions

TASK = "2027_wheelleg_walk"


class WheellegCfg:
    def __init__(self):
        self.__dict__.update(vars(FudanWalkCfg()))
        self.env.num_envs = 512
        self.env.play_num_envs = 20
        self.commands.velocity_initial = .5
        self.commands.velocity_max = 2.
        self.commands.stop_speed_range = [.25, .5]
        self.commands.yaw_range = [-1., 1.]
        self.commands.height_range = [.24, .28]
        self.commands.parking_heights = [.24, .26, .28]
        self.control.nominal_leg_positions = standing_motor_positions()
        self.control.max_wheel_velocity = 58.4
        self.domain_rand.nominal_offset_range = [-.02, .02]
        self.inference.checkpoint = None
        self.inference.checkpoint_sha256 = None


class WheellegCfgPPO:
    def __init__(self):
        self.__dict__.update(vars(FudanWalkCfgPPO()))
        self.seed = 1
        self.experiment_name = TASK
        self.run_name = ""


def config_record(cfg):
    import json
    result = {name: vars(section) for name, section in vars(cfg).items()
              if hasattr(section, "__dict__")}
    return json.loads(json.dumps(result))
