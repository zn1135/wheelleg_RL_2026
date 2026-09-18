"""从地面后摆初态起立的独立训练配置。"""

from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup_config import (
    ChuanliantuiStandupCfg,
    ChuanliantuiStandupCfgPPO,
)


class ChuanliantuiGroundStandupCfg(ChuanliantuiStandupCfg):
    class init_state(ChuanliantuiStandupCfg.init_state):
        # MuJoCo 中该虚拟姿态的无穿透贴地 root 高度约 0.147 m；取 0.15 m
        # 避免 Isaac 初始穿地。default_joint_angles 继承微蹲目标，不能设为后摆
        # 初态，否则策略的 ±0.5 rad 动作范围无法学回站姿。
        pos = [0.0, 0.0, 0.15]

    class standup(ChuanliantuiStandupCfg.standup):
        initial_base_height = 0.15
        # 训练 reset 的真实状态；DOF 顺序为 [lf0, lf1, lfwheel, rf0, rf1, rfwheel]。
        initial_dof_pos = [11.0, 0.0, 0.0, -11.0, 0.0, 0.0]

    class rewards(ChuanliantuiStandupCfg.rewards):
        class scales(ChuanliantuiStandupCfg.rewards.scales):
            # 强化从后摆低姿态向目标站高抬升的驱动力。
            base_height = 1.0
            # asset.penalize_contacts_on 只包含 base_link；其与地面的接触会触发此项。
            collision = -1.0
            orientation = -1.0

class ChuanliantuiGroundStandupCfgPPO(ChuanliantuiStandupCfgPPO):
    class runner(ChuanliantuiStandupCfgPPO.runner):
        experiment_name = "chuanliantui_ground_standup"
        max_iterations = 3000
        resume = False
