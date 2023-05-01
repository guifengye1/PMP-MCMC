import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from tqdm import tqdm
import copy
import math
from time import time

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



class HMCOptimizer():
    def __init__(self, net, alpha):
        super().__init__()
        self.net = net
        self.alpha = alpha
        self.d = sum(p.numel() for p in self.net.parameters())
        self.loss = torch.nn.CrossEntropyLoss().to(device)
        self.loss_list = []
        self.train_acc = []
        self.test_acc = []

    def step(self, s, path_len=0.001, step_size=0.1):
        # Step 1:
        proposal_net = copy.deepcopy(self.net)  # 复制样本，用来放迭代之后的样本
        p_old = torch.randn(self.d).to(device) * 0.0005  # 随机选择原速度
        p_new = copy.deepcopy(p_old).to(device)  # 随机选择新速度
        # 计算势能 U(x)=-log(p(x))
        yhat = self.net(X)
        x_0_nlp = - self.loss(yhat, y)
        # 计算动能
        p_0_nlp = (p_old * p_old).sum() / 2
        H_0 = p_0_nlp + x_0_nlp

        x_0_nlp.backward()
        du_dx = torch.tensor([]).to(device)
        for i in self.net.parameters():
            i = i.grad.reshape(-1)
            shape = i.shape[0]
            du_dx = torch.cat([du_dx, i.reshape(shape, 1)])

        du_dx = du_dx.reshape(self.d)
        # leapfrog 动力学迭代
        # 1. P走半步
        p_new += step_size * du_dx / 2  # as potential energy increases, kinetic energy

        # 2. 参数走一步
        sum = 0
        for i, j in zip(proposal_net.parameters(), range(self.d)):
            size = i.numel()
            i.data += step_size * p_new[sum:sum + size].reshape(i.data.shape)
            sum += size
        # 3.更新下半步所需要的参数
        yhat = proposal_net(X)
        x_1_nlp = - self.loss(yhat, y)
        x_1_nlp.backward()
        du_dx = torch.tensor([]).to(device)
        for i in proposal_net.parameters():
            i = i.grad.reshape(-1)
            shape = i.shape[0]
            du_dx = torch.cat([du_dx, i.reshape(shape, 1)])
        # 4. 走下半步
        du_dx = du_dx.reshape(self.d)
        p_new += step_size * du_dx.reshape(self.d) / 2  # second half-step "leapfrog" update to momentum

        p_1_nlp = (p_new * p_new).sum() / 2
        yhat = proposal_net(X)
        x_1_nlp = - self.loss(yhat, y)

        H_1 = x_1_nlp + p_1_nlp
        # print("建议：",-H_1,"旧的：",-H_0)
        acceptance = torch.exp((- H_0 + H_1) * 1000)
        rand = torch.rand(1)[0]
        # print("接受率: ",acceptance)
        if acceptance > rand:
            self.net = proposal_net
            self.loss_list.append(-x_1_nlp.data)
        else:
            self.loss_list.append(- x_0_nlp.data)
        if s % 200 == 0:
            print("loss = ",-x_1_nlp.data)
        correct = (yhat.argmax(1) == y).type(torch.float).sum().item()
        self.train_acc.append(correct / 50000 )
        correct = (proposal_net(x_test).argmax(1) == y_test).type(torch.float).sum().item()
        self.test_acc.append(correct / 10000 )
       
        # return self.net

    def fit(self, data=None, num_steps=1000):
        # We create one tensor per parameter so that we can keep track of the parameter values over time:

        for s in range(num_steps):
            self.step(s)
        return np.array(self.loss_list), np.array(self.train_acc), np.array(self.test_acc)


network = LeNet()

network.load_state_dict(torch.load("cifar.pkl"))
network = network.to(device)
trainer = HMCOptimizer(network, alpha=alpha)
loss,train_acc,test_acc = trainer.fit(num_steps=num_steps)
np.save('cifar_hmc_loss'+'.npy', loss)
np.save('cifar_hmc_train_acc'+'.npy', train_acc)
np.save('cifar_hmc_test_acc'+'.npy', test_acc)


