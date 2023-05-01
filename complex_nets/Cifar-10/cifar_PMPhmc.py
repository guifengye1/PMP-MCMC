import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 0.001
print("device: ", device)
num_steps = 10
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
class PMPHMCOptimizer():
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

    def step(self, s, p_s, nets_loss,tree_deep):
        A = torch.ones([self.N + 1, 1]).to(device)



        # 计算接受率（可并行）
        for all in range(self.N + 1):
            for c in range(int(tree_deep)):
                judg = all
                j = int(math.pow(2, c + 1))
                half_j = int(j / 2)
                if judg > 0:
                    deep = int(math.log2(judg))
                while (judg > j - 1):
                    if judg >= int(math.pow(2, deep)):
                        judg = judg - int(math.pow(2, deep))
                    deep = deep - 1
                if (judg < half_j):
                    w_new = torch.exp(nets_loss[judg] - (p_s[judg][judg + half_j]*p_s[judg][judg + half_j]).sum()/2)
                    w_old = torch.exp(nets_loss[judg+half_j] - (p_s[judg+ half_j][judg ]*p_s[judg+ half_j][judg] ).sum()/2)
                    A[all, 0] = A[all, 0] * max(torch.tensor(0),1-w_old/w_new)

                else:
                    w_new = torch.exp(nets_loss[judg] - (p_s[judg][judg - half_j]*p_s[judg][judg - half_j]).sum()/2)
                    w_old =torch.exp(nets_loss[judg-half_j] - (p_s[judg- half_j][judg ]*p_s[judg- half_j][judg] ).sum()/2)
                    A[all, 0] = A[all, 0] * min(torch.tensor(1), w_new / w_old)

        B = A.reshape(-1)

        B = torch.where(torch.isnan(B), torch.full_like(B, 1), B)
        B = torch.where(torch.isinf(B), torch.full_like(B, 1), B)
        I = torch.multinomial(B, 1, replacement=True).data
        return I


        # return self.net

    def fit(self, data=None, num_steps=1000,step_size=0.1):
        # We create one tensor per parameter so that we can keep track of the parameter values over time:
        print("-------PMP HMC Optimizer------ ")
        tree_deep = math.log2(self.N + 1)
        ## 参数，网络数
        proposal_nets = [1] * (self.N + 1)#保存各个网络
        p_s = torch.empty(self.N+1,self.N+1,self.d).to(device)#保存网络的速度p:p[i][j]表示用于i to j 时， i 的速度
        nets_loss = [1] * (self.N + 1)#保存各个网络的loss

        for s in range(num_steps):
            proposal_nets[0] = copy.deepcopy(self.net).to(device)  # 初始化产生N个建议网络（可并行）

            for i in range(int(tree_deep)):
                j = int(math.pow(2, i))
                for k in range(int(j)):
                    p_s[k][k+j] = torch.randn(self.d).to(device) * 0.0005  # 随机选择原速度
                    proposal_nets[k+j] = copy.deepcopy(proposal_nets[k]).to(device)
                    p_s[k+j][k]= copy.deepcopy(p_s[k][k+j])

                    # 计算势能 U(x)=-log(p(x))
                    yhat = proposal_nets[k](X)
                    nets_loss[k] = - self.loss(yhat, y)
                    # 计算du_dx
                    nets_loss[k].backward()
                    du_dx = torch.tensor([]).to(device)
                    for par in proposal_nets[k].parameters():
                        par = par.grad.reshape(-1)
                        shape = par.shape[0]
                        du_dx = torch.cat([du_dx, par.reshape(shape, 1)])
                    du_dx = du_dx.reshape(self.d)
                    # 1. P走半步
                    p_s[k+j][k] +=  step_size * du_dx / 2
                    # 2. 参数走一步
                    sum = 0
                    for par in proposal_nets[k+j].parameters():
                        size = par.numel()
                        par.data += step_size * p_s[k+j][k][sum:sum + size].reshape(par.data.shape)
                        sum += size
                        # 3.更新下半步所需要的参数
                    yhat = proposal_nets[k+j](X)
                    nets_loss[k+j] = - self.loss(yhat, y)
                    nets_loss[k+j].backward()
                    du_dx = torch.tensor([]).to(device)
                    for par in proposal_nets[j+k].parameters():
                        par = par.grad.reshape(-1)
                        shape = par.shape[0]
                        du_dx = torch.cat([du_dx, par.reshape(shape, 1)])
                    # 4. 走下半步
                    du_dx = du_dx.reshape(self.d)
                    p_s[k+j][k] +=  step_size * du_dx / 2
            I = self.step(s,p_s,nets_loss,tree_deep)
            self.net = proposal_nets[I]
            print("loss = ",- nets_loss[I])
            correct = (self.net(X).argmax(1) == y).type(torch.float).sum().item()
            self.train_acc.append(correct / 50000)
            correct = (self.net(x_test).argmax(1) == y_test).type(torch.float).sum().item()
            self.test_acc.append(correct / 10000)
        return np.array(self.loss_list), np.array(self.train_acc), np.array(self.test_acc)



network = LeNet()
network.load_state_dict(torch.load("cifar.pkl"))
network = network.to(device)

trainer = PMPHMCOptimizer(network, alpha=alpha,N=7)
loss, train_acc, test_acc = trainer.fit(num_steps=num_steps)

np.save('cifar_PMPhmc_loss'+'.npy', loss)
np.save('cifar_PMPhmc_train_acc'+'.npy', train_acc)
np.save('cifar_PMPhmc_test_acc'+'.npy', test_acc)

