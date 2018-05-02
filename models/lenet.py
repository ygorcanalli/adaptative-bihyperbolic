'''LeNet in PyTorch.'''
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


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class AdaptativeActivationLayer(nn.Module):
    def __init__(self, in_size, out_size, names=['lambda', 'tau_1', 'tau_2']):
        super(AdaptativeActivationLayer, self).__init__()

        self.parameters = {}
        self.in_size = in_size
        self.out_size = out_size
        for name in self.names:
            self.parameters[name] = None

    def forward(self, input):
        size = input.size()
        for name, parameter in self.parameters.items():
            if parameter is None:
                if name == 'lambda':
                    self.parameters[name] = nn.Parameter(torch.ones(size))
                else:
                    self.parameters[name] = nn.Parameter(torch.Tensor(size))
                    _unidimensional_xavier_normal(self.parameters[name],
                                                 self.in_size,
                                                 self.out_size)
        out = _bi_hyperbolic(input, self.parameters['lambda'],
                             self.parameters['tau_1'],
                             self.parameters['tau_2'])
        return out
