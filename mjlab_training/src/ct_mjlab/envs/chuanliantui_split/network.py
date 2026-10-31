"""25/125 history actor; the existing estimator optimizer stays independent."""

import copy
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from ..chuanliantui_stand_v2.network import HistoryPPO
from .observations import ACTOR_DIM, HISTORY_LENGTH

WALK_ACTOR_STD_MIN = .2


class HistoryActor(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = MLP(ACTOR_DIM * HISTORY_LENGTH, 3, (128, 64), 'elu')

    def _get_latent_dim(self):
        return self.obs_dim + 3

    def get_latent(self, obs, masks=None, hidden_state=None):
        velocity = self.encoder(obs['history'])
        return torch.cat((super().get_latent(obs, masks, hidden_state), velocity.detach()), -1)

    def as_onnx(self, verbose=False):
        return HistoryPolicy(self)

    def as_jit(self):
        return HistoryPolicy(self)


class HistoryPolicy(torch.nn.Module):
    is_recurrent = False
    input_names = ['observation_history']
    output_names = ['actions', 'velocity']

    def __init__(self, actor):
        super().__init__()
        self.encoder, self.mlp = copy.deepcopy(actor.encoder), copy.deepcopy(actor.mlp)

    def forward(self, observation_history):
        velocity = self.encoder(observation_history)
        return self.mlp(torch.cat((observation_history[:, -ACTOR_DIM:], velocity), -1)), velocity / 2.

    def get_dummy_inputs(self):
        return (torch.zeros(1, ACTOR_DIM * HISTORY_LENGTH),)
