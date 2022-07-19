import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .utils import init

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))

init_tanh_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    """
    Flatten a tensor
    """
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )

class ResBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MixStyleLayer(nn.Module):
    """
    Style Randomization Layer
    reference: https://github.com/hyeonseobnam/sagnet
    """
    def __init__(self, eps=1e-5, attacker_log_std_min=-10, attacker_log_std_max=2):
        super(MixStyleLayer, self).__init__()
        self.eps = eps
        self.attacker_log_std_min = attacker_log_std_min
        self.attacker_log_std_max = attacker_log_std_max

    def forward(self, x):
        N, C, H, W = x.size()
        device = x.device

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            
            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)
            alpha = alpha.to(device)
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x

    def forward_adv(self, x, adv):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            adv = adv.view(N, C, 2)
            mu, log_std = adv[...,0, None], adv[...,1, None]
            log_std = self.attacker_log_std_min + 0.5 * ( \
                        self.attacker_log_std_max - self.attacker_log_std_min \
	    	            ) * (log_std + 1)
            
            x = x * (log_std.exp() + self.eps).sqrt() + mu
            x = x.view(N, C, H, W)

        return x
        
class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class MLPBase(NNBase):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        self.actor = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class ResNetBase(NNBase):
    """
    Residual Network 
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32], encoding=False):
        super(ResNetBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.encoding = encoding
        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(ResBlock(out_channels))
        layers.append(ResBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        if self.encoding:
            z = x.clone().detach()
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        if self.encoding:
            return self.critic_linear(x), x, rnn_hxs, z
        else:
            return self.critic_linear(x), x, rnn_hxs

class SarBase(NNBase):
    """
    Sar Encoder Network 
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32], n_envs=64, encoding=False):
        super(SarBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.mix = MixStyleLayer()
        self.attacker = self.build_attacker(n_envs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.encoding = encoding
        self.train()

    def build_attacker(self, n_envs, hidden_size=64):
        x = torch.zeros((n_envs, 3, 64, 64))
        with torch.no_grad():
            x = self.layer1(x)
            x = self.layer2(x)

        attacker = nn.Sequential(
			nn.ReLU(),
			Flatten(),
			nn.Linear(self.flatten(x).size()[-1], x.size()[1] * 2),
			nn.Tanh()
		    )
        return attacker

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(ResBlock(out_channels))
        layers.append(ResBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)

        adv = self.attacker(torch.detach(x))
        x_adv = self.mix.forward_adv(x, adv)
        if self.encoding:
            z = x.clone().detach()
            z_adv = x_adv.clone().detach()

        x = self.mix(x)
        x = self.layer3(x)
        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        x_adv = self.layer3(x_adv)
        x_adv = self.relu(self.flatten(x_adv))
        x_adv = self.relu(self.fc(x_adv))

        if self.is_recurrent:
            raise NotImplemented
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        if self.encoding:
            return self.critic_linear(x), self.critic_linear(x_adv), \
                    x, x_adv, rnn_hxs, z, z_adv
        else:
            return self.critic_linear(x), self.critic_linear(x_adv), \
                    x, x_adv, rnn_hxs
