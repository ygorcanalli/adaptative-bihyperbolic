from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.autograd import Variable

def bi_hyperbolic_fn(x, lmbda, tau_1, tau_2):
    return torch.sqrt(1/16*(4 * lmbda * x + 1)**2 + tau_1**2) - torch.sqrt(1/16*(4 * lmbda * x - 1)**2 + tau_2**2)
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc_in = nn.Linear(28*28, 800)
        self.fc_h1 = nn.Linear(800, 800)
        self.fc_h2 = nn.Linear(800, 800)
        self.fc_h3 = nn.Linear(800, 800)
        self.fc_h4 = nn.Linear(800, 800)
        self.fc_h5 = nn.Linear(800, 800)

        self.fc_out = nn.Linear(800, 10)
        
        self.lambdas = torch.ones(6, 800)
        self.taus_1 = torch.Tensor(6, 800)
        init.xavier_normal(self.taus_1)
        self.taus_2 = torch.Tensor(6, 800)
        init.xavier_normal(self.taus_2)

        self.lambdas = nn.Parameter(self.lambdas)
        self.taus_1 = nn.Parameter(self.taus_1)
        self.taus_2 = nn.Parameter(self.taus_2)

    def forward(self, x):
        x = x.view(-1, 28*28)
        
        x = bi_hyperbolic_fn(self.fc_in(x),
                             self.lambdas[0],
                             self.taus_1[0],
                             self.taus_2[0])
        x = bi_hyperbolic_fn(self.fc_h1(x),
                             self.lambdas[1],
                             self.taus_1[1],
                             self.taus_2[1])
        x = bi_hyperbolic_fn(self.fc_h2(x),
                             self.lambdas[2],
                             self.taus_1[2],
                             self.taus_2[2])
        x = bi_hyperbolic_fn(self.fc_h3(x),
                             self.lambdas[3],
                             self.taus_1[3],
                             self.taus_2[3])
        x = bi_hyperbolic_fn(self.fc_h4(x),
                             self.lambdas[4],
                             self.taus_1[4],
                             self.taus_2[4])
        x = bi_hyperbolic_fn(self.fc_h5(x),
                             self.lambdas[5],
                             self.taus_1[5],
                             self.taus_2[5])
        x = F.relu(self.fc_out(x))
        return F.log_softmax(x, dim=1)

model = MLPNet()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
