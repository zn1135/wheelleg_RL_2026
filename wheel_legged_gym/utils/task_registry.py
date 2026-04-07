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

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
from shutil import copyfile
import ntpath

from wheel_legged_gym.rsl_rl.env.vec_env import VecEnv
from wheel_legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    get_load_path,
    set_seed,
    parse_sim_params,
)
from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(
        self,
        name: str,
        task_class: VecEnv,
        env_cfg: LeggedRobotCfg,
        train_cfg: LeggedRobotCfgPPO,
    ):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def save_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """将当前训练相关配置与关键环境脚本备份到本次日志目录。"""
        # 创建本次运行的日志目录
        os.mkdir(self.log_dir)

        # 需要备份的基础文件与任务配置文件
        save_items = [
            os.path.join(
                self.log_dir,
                WHEEL_LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py",
            ),
            os.path.join(
                self.log_dir,
                WHEEL_LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py",
            ),
            os.path.join(
                self.log_dir,
                WHEEL_LEGGED_GYM_ENVS_DIR
                + "/{}/".format(name)
                + "{}_config.py".format(name),
            ),
        ]
        # 某些任务有同名实现文件（如 xxx.py），存在时一并备份
        py_root = os.path.join(
            WHEEL_LEGGED_GYM_ENVS_DIR + "/{}/".format(name) + "{}.py".format(name),
        )
        if os.path.exists(py_root):
            save_items.append(os.path.join(self.log_dir, py_root))
        # 将待保存文件复制到日志目录（仅保留文件名）
        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self.log_dir + "/" + base_file_name)

    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """创建环境实例（可从已注册任务名加载，也可使用传入配置覆盖）。

        参数:
            name (string): 已注册环境任务名。
            args (Args, optional): Isaac Gym 命令行参数；为 None 时调用 get_args() 获取。
            env_cfg (Dict, optional): 传入的环境配置；为 None 时使用注册表中的默认配置。

        异常:
            ValueError: 当 name 未在任务注册表中找到时抛出。

        返回:
            isaacgym.VecTaskPython: 创建好的环境实例。
            Dict: 对应的环境配置。
        """
        # 未显式传入 args 时，读取命令行参数
        if args is None:
            args = get_args()
        # 检查任务名是否已注册，并获取对应环境类
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # 未传入环境配置时，从注册表读取默认配置
            env_cfg, _ = self.get_cfgs(name)
        # 用命令行参数覆盖配置中的可覆盖项
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        # 设置随机种子，保证实验可复现
        set_seed(env_cfg.seed)
        # 构造并解析仿真参数（先将配置对象转为字典）
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        # 实例化具体任务环境
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env, env_cfg

    def make_alg_runner(
        self, env, name=None, args=None, train_cfg=None, log_root="default"
    ) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """创建训练算法运行器（可从注册任务加载，或使用传入训练配置）。

        参数:
            env (isaacgym.VecTaskPython): 用于训练的环境实例。
            name (string, optional): 已注册环境任务名；当 train_cfg 为 None 时用于读取配置。
            args (Args, optional): Isaac Gym 命令行参数；为 None 时调用 get_args() 获取。
            train_cfg (Dict, optional): 训练配置；为 None 时从 name 对应注册配置加载。
            log_root (str, optional): Tensorboard 日志根目录。
                                      设为 None 表示不记录日志；
                                      设为 "default" 时默认保存到 <ROOT>/logs/<experiment_name>/。

        异常:
            ValueError: 当 name 与 train_cfg 同时为空时抛出。

        返回:
            PPO: 创建好的算法运行器。
            Dict: 对应训练配置。
        """
        # 未显式传入 args 时，读取命令行参数
        if args is None:
            args = get_args()
        # 优先使用传入训练配置；否则根据任务名从注册表读取
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # 从注册表加载训练配置
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # 使用命令行参数覆盖训练配置中的可覆盖项
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 解析日志目录：默认路径 / 不记录 / 自定义路径
        if log_root == "default":
            log_root = os.path.join(
                WHEEL_LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
            )
            if not os.path.exists(log_root):
                os.mkdir(log_root)
            # 日志目录格式：时间戳 + run_name + exptid
            self.log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg.runner.run_name
                + args.exptid,
            )
        elif log_root is None:
            self.log_dir = None
        else:
            # 自定义根目录下同样按时间戳组织一次运行
            self.log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg.runner.run_name
                + args.exptid,
            )

        # 训练器接收字典配置，因此先做对象到字典的转换
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(
            env, train_cfg_dict, self.log_dir, device=args.rl_device
        )
        # 若配置要求续训，则从历史权重恢复
        resume = train_cfg.runner.resume
        if resume:
            # 根据 run/checkpoint 定位待加载模型路径
            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg


# make global task registry
task_registry = TaskRegistry()
