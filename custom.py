from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _unidimensional_xavier_normal(tensor, fan_in, fan_out, gain=1):
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with torch.autograd.no_grad():
        return tensor.normal_(0, std)

def _bi_hyperbolic(tensor, lmbda, tau_1, tau_2):
    return (torch.sqrt(1/16*(4 * lmbda * tensor + 1)**2 + tau_1**2) -
            torch.sqrt(1/16*(4 * lmbda * tensor - 1)**2 + tau_2**2))

class MLPNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MLPNet, self).__init__()
        self.fc_hiddens = []

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.fc_in = nn.Linear(in_size, hidden_sizes[0])

        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.fc_hiddens.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.fc_hiddens.append(nn.Linear(hidden_sizes[i-1], hidden_size))
        self.fc_hiddens = nn.ModuleList(self.fc_hiddens)
        self.fc_out = nn.Linear(hidden_sizes[-1], out_size)

    def forward(self, input):
        output = input.view(-1, self.in_size)
        output = F.relu(self.fc_in(output))
        for i in range(len(self.fc_hiddens)):
            output = F.relu(self.fc_hiddens[i](output))
        output = self.fc_out(output)
        return F.log_softmax(output, dim=1)

class AdaptativeBiHyperbolicLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(AdaptativeBiHyperbolicLayer, self).__init__()

        self._lambda = nn.Parameter(torch.ones(out_size))
        self._tau_1 = nn.Parameter(torch.Tensor(out_size))
        _unidimensional_xavier_normal(self._tau_1,
            in_size, out_size)
        self._tau_2 = nn.Parameter(torch.Tensor(out_size))
        _unidimensional_xavier_normal(self._tau_2,
            in_size, out_size)

    def forward(self, input):
        return _bi_hyperbolic(input, self._lambda,
                              self._tau_1, self._tau_2)

class AdaptativeBiHyperbolicMLPNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes):
        super(AdaptativeBiHyperbolicMLPNet, self).__init__()
        self.layers = []

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.layers.append(nn.Linear(in_size, hidden_sizes[0]))
        self.layers.append(AdaptativeBiHyperbolicLayer(
            in_size, hidden_sizes[0]))

        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                self.layers.append(AdaptativeBiHyperbolicLayer(
                    hidden_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))
                self.layers.append(AdaptativeBiHyperbolicLayer(
                    hidden_sizes[i-1], hidden_size))

        self.layers.append(nn.Linear(hidden_sizes[-1], out_size))
        self.layers.append(AdaptativeBiHyperbolicLayer(
            hidden_sizes[-1], out_size))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, input):
        output = input.view(-1, self.in_size)
        for layer in self.layers:
            output = layer(output)

        return F.log_softmax(output, dim=1)
