"""Velocity-supervised history encoder; PPO keeps the encoder detached like Isaac."""

import copy

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

from .observations import ACTOR_DIM, HISTORY_LENGTH


class HistoryActor(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = MLP(ACTOR_DIM * HISTORY_LENGTH, 3, (128, 64), "elu")

    def _get_latent_dim(self):
        return self.obs_dim + 3

    def get_latent(self, obs, masks=None, hidden_state=None):
        velocity = self.encoder(obs["history"])
        return torch.cat((super().get_latent(obs, masks, hidden_state), velocity.detach()), -1)

    def as_onnx(self, verbose=False):
        return HistoryPolicy(self, verbose)

    def as_jit(self):
        return HistoryPolicy(self)


class HistoryPolicy(torch.nn.Module):
    """Stateless deployment graph: history includes the current observation last."""
    is_recurrent = False
    input_names = ["observation_history"]
    output_names = ["actions", "velocity"]

    def __init__(self, actor, verbose=False):
        super().__init__()
        self.encoder = copy.deepcopy(actor.encoder)
        self.mlp = copy.deepcopy(actor.mlp)
        self.verbose = verbose

    def forward(self, observation_history):
        # The newest frame is shared by the actor and estimator, without a second input.
        observations = observation_history[:, -ACTOR_DIM:]
        velocity = self.encoder(observation_history)
        return self.mlp(torch.cat((observations, velocity), -1)), velocity / 2.

    def get_dummy_inputs(self):
        return (torch.zeros(1, ACTOR_DIM * HISTORY_LENGTH),)


class HistoryPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Separate optimizers: PPO must not accidentally change the estimator.
        encoder_ids = {id(p) for p in self._raw_actor.encoder.parameters()}
        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in encoder_ids]
        self.encoder_optimizer = torch.optim.Adam(self._raw_actor.encoder.parameters(), lr=1e-3)

    def update(self):
        # Base PPO clears the buffer's cursor, not its tensors; preserve input before reuse.
        samples = self.storage.observations.flatten(0, 1)
        history, target = samples["history"].detach(), samples["velocity_target"].detach()
        # PPO clips all actor gradients, so remove stale estimator gradients first.
        self.encoder_optimizer.zero_grad(set_to_none=True)
        losses = super().update()
        total, count = 0., 0
        for _ in range(self.num_learning_epochs):
            for ids in torch.randperm(len(history), device=history.device).tensor_split(self.num_mini_batches):
                if not len(ids):
                    continue
                loss = (self._raw_actor.encoder(history[ids]) - target[ids]).square().mean()
                self.encoder_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._raw_actor.encoder.parameters(), .1)
                self.encoder_optimizer.step()
                total += float(loss.detach())
                count += 1
        losses["velocity_estimation"] = total / count
        return losses

    def save(self):
        return {**super().save(), "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict()}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        restored = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer", False):
            self.encoder_optimizer.load_state_dict(loaded_dict["encoder_optimizer_state_dict"])
        return restored
