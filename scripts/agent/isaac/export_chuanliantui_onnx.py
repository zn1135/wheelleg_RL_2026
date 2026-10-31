#!/usr/bin/env python3
"""导出完整 chuanliantui ActorCriticSequence 推理策略为 ONNX。

输入为当前 25 维 actor 观测和按 FIFO 排列的 125 维历史观测；输出为确定性
动作均值和 encoder latent。该脚本明确不使用 policy_1.pt，因为它只含 actor、
缺少 encoder。
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from sim2sim.mj_sim2sim_ct import (
    LATENT_DIM,
    NUM_ACTIONS,
    NUM_ENCODER_OBS,
    NUM_OBS,
    load_policy,
)


class SequencePolicyONNX(nn.Module):
    """将 encoder 和 actor 合为无状态的确定性 ONNX 推理图。"""

    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = actor_critic.encoder
        self.actor = actor_critic.actor

    def forward(self, observations, observation_history):
        latent = self.encoder(observation_history)
        actions = self.actor(torch.cat((observations, latent), dim=-1))
        return actions, latent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="完整 model_*.pt checkpoint；不可使用仅含 actor 的 policy_1.pt。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="导出的 .onnx 文件路径。",
    )
    parser.add_argument(
        "--fixed-batch", action="store_true",
        help="固定 batch=1，供 STM32Cube.AI 等嵌入式工具使用。",
    )
    parser.add_argument(
        "--opset", type=int, choices=(13, 15, 17), default=17,
        help="ONNX opset（默认 17）；按目标 Cube.AI 版本选择。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    if checkpoint.suffix != ".pt":
        raise ValueError(f"checkpoint 必须是 .pt 文件: {checkpoint}")

    actor_critic = load_policy(str(checkpoint), device="cpu")
    policy = SequencePolicyONNX(actor_critic).cpu().eval()
    observations = torch.zeros(1, NUM_OBS, dtype=torch.float32)
    observation_history = torch.zeros(1, NUM_ENCODER_OBS, dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        policy,
        (observations, observation_history),
        str(output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observations", "observation_history"],
        output_names=["actions", "latent"],
        dynamic_axes=None if args.fixed_batch else {
            "observations": {0: "batch"},
            "observation_history": {0: "batch"},
            "actions": {0: "batch"},
            "latent": {0: "batch"},
        },
    )

    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    batch = "1" if args.fixed_batch else "batch"
    metadata = {
        "architecture": "ActorCriticSequence",
        "observation_layout": "25d chuanliantui actor observation",
        "history_layout": "5 x 25 FIFO, oldest-to-newest",
        "observations_shape": f"{batch} x {NUM_OBS}",
        "observation_history_shape": f"{batch} x {NUM_ENCODER_OBS}",
        "actions_shape": f"{batch} x {NUM_ACTIONS}",
        "latent_shape": f"{batch} x {LATENT_DIM}",
        "opset": str(args.opset),
        "checkpoint": str(checkpoint),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(output))

    # 不依赖 onnxruntime；用 ONNX 自带的参考执行器与 PyTorch 导出源逐元素核对。
    from onnx.reference import ReferenceEvaluator

    torch.manual_seed(0)
    observations = torch.randn(1, NUM_OBS, dtype=torch.float32)
    observation_history = torch.randn(1, NUM_ENCODER_OBS, dtype=torch.float32)
    with torch.no_grad():
        torch_actions, torch_latent = policy(observations, observation_history)
    onnx_actions, onnx_latent = ReferenceEvaluator(model).run(
        None,
        {
            "observations": observations.numpy(),
            "observation_history": observation_history.numpy(),
        },
    )
    np.testing.assert_allclose(
        onnx_actions, torch_actions.numpy(), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        onnx_latent, torch_latent.numpy(), rtol=1e-5, atol=1e-6
    )

    print(f"ONNX 导出并校验通过: {output}")
    print(
        f"输入: observations=[{batch},25], observation_history=[{batch},125]; "
        f"输出: actions=[{batch},6], latent=[{batch},3]；与 PyTorch 数值对比通过"
    )


if __name__ == "__main__":
    main()
