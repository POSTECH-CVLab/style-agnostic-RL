import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import *
from .distributions import Categorical, SoftCategrocial

class PpoPolicy(nn.Module):
    """
    Actor-Critic module 
    """
    def __init__(self, obs_shape, num_actions, base_kwargs=None, encoding=False):
        super(PpoPolicy, self).__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        if len(obs_shape) == 3:
            base = ResNetBase
        elif len(obs_shape) == 1:
            base = MLPBase

        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = SoftCategrocial(self.base.output_size, num_actions)
        self.encoding = encoding
        
    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.encoding: 
            value, actor_features, rnn_hxs, z = self.base(inputs, rnn_hxs, masks)
            dist, logits = self.dist(actor_features)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()
            
            return value, action, action_log_probs, logits, rnn_hxs, z
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
            dist, logits = self.dist(actor_features)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

            return value, action, action_log_probs, logits, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist, logits = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, logits, rnn_hxs

class SarPolicy(nn.Module):
    """
    Style Agnostic Policy
    """
    def __init__(self, obs_shape, num_actions, base_kwargs=None, encoding=False):
        super(SarPolicy, self).__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        if len(obs_shape) == 3:
            base = SarBase
        else:
            raise NotImplemented

        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = SoftCategrocial(self.base.output_size, num_actions)
        self.encoding = encoding
        
    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.encoding: 
            value, value_adv, \
                features, features_adv, rnn_hxs, z, z_adv = self.base(inputs, rnn_hxs, masks)
            dist, logits = self.dist(features)
            _, logits_adv = self.dist(features_adv)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

            return value, action, action_log_probs, logits, rnn_hxs, \
                    value_adv, logits_adv, z, z_adv
        else:
            value, value_adv, \
                features, features_adv, rnn_hxs = self.base(inputs, rnn_hxs, masks)
            dist, logits = self.dist(features)
            _, logits_adv = self.dist(features_adv)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

            return value, action, action_log_probs, logits, rnn_hxs, \
                    value_adv, logits_adv

    def get_value(self, inputs, rnn_hxs, masks):
        value, value_adv, _, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, value_adv, \
            features, features_adv, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist, logits = self.dist(features)
        _, logits_adv = self.dist(features_adv)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, logits, rnn_hxs, \
                value_adv, logits_adv

    def act_parameters(self):
        params = []
        params += [p for p in self.dist.parameters()]
        params += [p for p in self.base.layer1.parameters()]
        params += [p for p in self.base.layer2.parameters()]
        params += [p for p in self.base.layer3.parameters()]
        params += [p for p in self.base.fc.parameters()]
        params += [p for p in self.base.critic_linear.parameters()]

        return params

    def adv_parameters(self):
        params = []
        params += [p for p in self.base.attacker.parameters()]

        return params



