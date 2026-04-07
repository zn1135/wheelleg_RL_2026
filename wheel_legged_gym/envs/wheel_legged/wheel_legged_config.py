# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class WheelLeggedCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.25]  # 机器人初始基座位置 x,y,z [m]
        default_joint_angles = {  # 动作为 0.0 时各关节的目标角度
            "lf0_Joint": 0.5,
            "lf1_Joint": 0.35,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": -0.5,
            "rf1_Joint": -0.35,
            "r_wheel_Joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # 位置/速度动作缩放系数，用于将策略输出映射到控制目标
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD 驱动参数：刚度决定“拉回目标位置”的强度
        stiffness = {"f0": 40.0, "f1": 40.0, "wheel": 0}  # [N*m/rad]
        # PD 驱动参数：阻尼抑制振荡，提升控制稳定性
        damping = {"f0": 1.0, "f1": 1.0, "wheel": 0.5}  # [N*m*s/rad]

    class asset(LeggedRobotCfg.asset):
        # 机器人模型与基础几何参数
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/wl/urdf/wl.urdf"
        name = "WheelLegged"
        offset = 0.054  # 机体几何偏置参数
        l1 = 0.15  # 连杆长度参数 1
        l2 = 0.25  # 连杆长度参数 2
        penalize_contacts_on = ["lf", "rf", "base"]  # 这些部位接触会被加入惩罚
        terminate_after_contacts_on = ["base"]  # 这些部位发生接触后终止回合
        self_collisions = 1  # 自碰撞开关：1 关闭，0 开启（位掩码过滤）
        flip_visual_attachments = False


class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # 日志实验名
        experiment_name = "wheel_legged"
