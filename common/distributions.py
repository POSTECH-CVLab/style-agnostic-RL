import math

import torch
import torch.nn as nn

from .utils import init

class FixedCategorical(torch.distributions.Categorical):
    """
    Categorical distribution object
    """
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

class Categorical(nn.Module):
    """
    Categorical distribution (NN module)
    """
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x), x

class SoftCategrocial(nn.Module):
    """
    Categorical distribution (NN module)
    """
    def __init__(self, num_inputs, num_outputs):
        super(SoftCategrocial, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return FixedCategorical(probs=x), x