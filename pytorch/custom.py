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
    return (math.sqrt(1/16*(4 * lmbda * tensor + 1)**2 + tau_1**2) -
            math.sqrt(1/16*(4 * lmbda * tensor - 1)**2 + tau_2**2))

class AdaptativeBiHyperbolicMLPNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes):
        super(AdaptativeBiHyperbolicMLPNet, self).__init__()
        self.fc_hiddens = []
        self.lambdas = []
        self.taus_1 = []
        self.taus_2 = []

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.fc_in = nn.Linear(in_size, hidden_sizes[0])
        alpha = torch.ones(hidden_sizes[0])
        tau_1 = torch.Tensor(hidden_sizes[0])
        tau_2 = torch.Tensor(hidden_sizes[0])
        _unidimensional_xavier_normal(tau_1, in_size, hidden_sizes[0])
        _unidimensional_xavier_normal(tau_2, in_size, hidden_sizes[0])

        self.lambdas.append(nn.Parameter(alpha))
        self.taus_1.append(nn.Parameter(tau_1))
        self.taus_2.append(nn.Parameter(tau_2))

        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.fc_hiddens.append(nn.Linear(hidden_size, hidden_size))
                alpha = torch.ones(hidden_size)
                tau_1 = torch.Tensor(hidden_size)
                tau_2 = torch.Tensor(hidden_size)
                _unidimensional_xavier_normal(tau_1, hidden_size, hidden_size)
                _unidimensional_xavier_normal(tau_2, hidden_size, hidden_size)

                self.lambdas.append(nn.Parameter(alpha))
                self.taus_1.append(nn.Parameter(tau_1))
                self.taus_2.append(nn.Parameter(tau_2))

            else:
                self.fc_hiddens.append(nn.Linear(hidden_sizes[i-1], hidden_size))
                alpha = torch.ones(hidden_size)
                tau_1 = torch.Tensor(hidden_size)
                tau_2 = torch.Tensor(hidden_size)
                _unidimensional_xavier_normal(tau_1, hidden_sizes[i-1], hidden_size)
                _unidimensional_xavier_normal(tau_2, hidden_sizes[i-1], hidden_size)

                self.lambdas.append(nn.Parameter(alpha))
                self.taus_1.append(nn.Parameter(tau_1))
                self.taus_2.append(nn.Parameter(tau_2))
        self.fc_out = nn.Linear(hidden_sizes[-1], out_size)

    def forward(self, input):
        output = input.view(-1, 28*28)
        output = _bi_hyperbolic(self.fc_in(output), self.lambdas[0],
                                self.taus_1[0], self.taus_2[0])
        for i in range(len(self.fc_hiddens)):
            output = _bi_hyperbolic(self.fc_hiddens[i](output), self.lambdas[i+1],
                                    self.taus_1[i+1], self.taus_2[i+1])
        output = self.fc_out(output)
        return F.log_softmax(output, dim=1)
