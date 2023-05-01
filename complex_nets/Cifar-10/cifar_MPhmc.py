import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn

# device = 'cuda:3' if torch.cuda.is_available() else 'cpu'   0.0005
# device = 'cuda:7' if torch.cuda.is_available() else 'cpu'    0.0001
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 0.001
print("device: ", device)
num_steps = 100000
alpha = 0.001
# get data
transform = torchvision.transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

X = torch.tensor(train_dataset.data, dtype=torch.float).to(device)
X = X.permute(0, 3, 1, 2)

y = torch.tensor(train_dataset.targets).to(device)
x_test = torch.tensor(test_dataset.data, dtype=torch.float).to(device)
x_test = x_test.permute(0, 3, 1, 2)
y_test = torch.tensor(test_dataset.targets).to(device)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)


import numpy as np
from tqdm import tqdm
import copy
import math
from time import time

import random
class MPHMCOptimizer():
    def __init__(self, net, alpha,N):
        super().__init__()
        self.net = net
        self.N = N
        self.alpha = alpha
        self.d = sum(p.numel() for p in self.net.parameters())
        self.loss = torch.nn.CrossEntropyLoss().to(device)
        self.loss_list = []
        self.train_acc = []
        self.test_acc = []

    def step(self, s, p_s, nets_loss):
        A = torch.zeros([self.N + 1, 1]).to(device)
        # 1.计算接受权重
        for j in range(1,self.N+1):
            A[j,0] = torch.exp(min(A[j,0],nets_loss[j] - (p_s[j]*p_s[j]).sum()/2-nets_loss[0] + (p_s[0]*p_s[0]).sum()/2 ))
        A[0,0] = (self.N) - A.sum()
        B = A.reshape(-1)
        B = torch.where(torch.isnan(B), torch.full_like(B, 1), B)
        B = torch.where(torch.isinf(B), torch.full_like(B, 1), B)
        I = torch.multinomial(B, 1, replacement=True).data
        return I

    def fit(self, data=None, num_steps=1000,step_size=0.1):

        print("-------MP HMC Optimizer------ ")
        ## 参数，网络数
        proposal_nets = [1] * (self.N + 1)#保存各个网络
        p_s = [1] * (self.N + 1)#保存网络的速度p
        nets_loss = [1] * (self.N + 1)#保存各个网络的loss

        for s in range(num_steps):
            proposal_nets[0] = copy.deepcopy(self.net).to(device)  # 初始化产生N个建议网络（可并行）
            p_s[0] = torch.randn(self.d).to(device) * 0.0005  # 随机选择原速度


            ranint = int(random.uniform(1, self.N + 1))
            sign = 1.0
            for i in range(self.N):
                if i>=ranint:
                    sign = -1.0
                j = i + 1
                proposal_nets[j] = copy.deepcopy(proposal_nets[i]).to(device)
                p_s[j] = copy.deepcopy(p_s[i])
                # 计算势能 U(x)=-log(p(x))
                yhat = proposal_nets[i](X)
                nets_loss[i] = - self.loss(yhat, y)
                # 计算du_dx
                nets_loss[i].backward()
                du_dx = torch.tensor([]).to(device)
                for i in proposal_nets[i].parameters():
                    i = i.grad.reshape(-1)
                    shape = i.shape[0]
                    du_dx = torch.cat([du_dx, i.reshape(shape, 1)])
                du_dx = du_dx.reshape(self.d)
                # 1. P走半步
                p_s[j] += sign * step_size * du_dx / 2
                # 2. 参数走一步
                sum = 0
                for par in proposal_nets[j].parameters():
                    size = par.numel()
                    par.data += sign * step_size * p_s[j][sum:sum + size].reshape(par.data.shape)
                    sum += size
                # 3.更新下半步所需要的参数
                yhat = proposal_nets[j](X)
                nets_loss[j] = - self.loss(yhat, y)
                nets_loss[j].backward()
                du_dx = torch.tensor([]).to(device)
                for par in proposal_nets[j].parameters():
                    par = par.grad.reshape(-1)
                    shape = par.shape[0]
                    du_dx = torch.cat([du_dx, par.reshape(shape, 1)])
                # 4. 走下半步
                du_dx = du_dx.reshape(self.d)
                p_s[j] += sign * step_size * du_dx / 2

            I = self.step(s,p_s,nets_loss)

            self.net = proposal_nets[I]
            print("loss = ",- nets_loss[I])
            correct = (self.net(X).argmax(1) == y).type(torch.float).sum().item()
            self.train_acc.append(correct / 50000 )
            correct = (self.net(x_test).argmax(1) == y_test).type(torch.float).sum().item()
            self.test_acc.append(correct / 10000 )
        return np.array(self.loss_list), np.array(self.train_acc), np.array(self.test_acc)



network = LeNet()
network.load_state_dict(torch.load("cifar.pkl"))
network = network.to(device)

trainer = MPHMCOptimizer(network, alpha=alpha, N=3)
loss, train_acc, test_acc = trainer.fit(num_steps=num_steps)

np.save('cifar_MPhmc_loss'+'.npy', loss)
np.save('cifar_MPhmc_train_acc'+'.npy', train_acc)
np.save('cifar_MPhmc_test_acc'+'.npy', test_acc)