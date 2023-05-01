import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import copy
import numpy as np
from matplotlib import pyplot as plt
import math

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
number_MP = 250000
alpha = 0.0001
N = 7
# 定义超参数
batch_size = 60000  # 一次训练的样本数目


class Model(torch.nn.Module):
    # 构建模型（简单的卷积神经网络）
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 10, 24x24
        self.conv2 = nn.Conv2d(10, 20, 3)  # 128, 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)  # 展开成一维，方便进行FC
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


# 2. 数据获取
def get_data():
    # 获取训练集
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
    train_loader = DataLoader(train, batch_size=batch_size)  # 分割训练集

    # 获取测试集
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loader = DataLoader(test, batch_size=batch_size)  # 分割测试集

    # 返回分割好的训练集和测试集
    return train_loader, test_loader


train_loader, test_loader = get_data()
for _, (X, y) in enumerate(train_loader):
    X = X.to(device)
    y = y.to(device)
for _, (x_test,y_test) in enumerate(test_loader):
    x_test = x_test.to(device)
    y_test = y_test.to(device)


@torch.no_grad()
def loss(net):
    yhat = net(X)
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(yhat, y) / 10


class MPOptimizer():
    def __init__(self, net, alpha):
        super().__init__()
        self.net = net
        self.alpha = alpha
        self.first = 0
        self.loss = None
        self.loss_proposal = None
        self.lamb = 10000
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.N = 7  # parallel number
        self.d = sum(p.numel() for p in self.net.parameters())  # 参数维度为d
        self.sigma = 1
        self.const_1 = torch.log(torch.sqrt(torch.tensor(1 / (2 * np.pi))) / self.sigma).to(device)
        self.const_2 = torch.tensor(1 / 2 / self.sigma ** 2).to(device)
        self.loss_list = []

    @torch.no_grad()
    def update(self, net):  # update net
        my_net = copy.deepcopy(net)
        for name, par in my_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)
        return my_net

    @torch.no_grad()
    def step(self, s, proposal_nets=None, proposal_nets_paras=None, para_num=None):
        # Step 1: calculate accept weight
        A = torch.empty([self.N + 1, 1]).to(device)
        tran = torch.zeros([self.N + 1, self.N + 1]).to(device)
        for j in range(self.N + 1):
            for k in range(j + 1, self.N + 1):
                tran[j][k] = (self.const_1 - (
                            proposal_nets_paras[j] - proposal_nets_paras[k]) ** 2 * self.const_2).sum() / para_num
                tran[k][j] = tran[j][k]
        for j in range(self.N + 1):
            temp = tran[j].sum()
            A[j, 0] = temp / (self.N + 1) - loss(proposal_nets[j])
        # deal weights
        mean_a = torch.mean(A)
        std_a = torch.std(A)
        A = (A - mean_a) / std_a
        B = torch.exp(A).reshape(-1)
        # resample
        I = torch.multinomial(B, 1, replacement=True).data
        self.net = proposal_nets[I]
        # logging loss
        pred = self.net(X)
        self.loss = self.loss_fn(pred, y).item()
        self.loss_list.append(self.loss)

        if s % 1000 == 0:
            print("loss = ", self.loss)
            if s % 10000 == 0:
                correct = (pred.argmax(1) == y).type(torch.float).sum().item()
                print("train correct ratio = ", correct / batch_size * 100, "%")

                pred = self.net(x_test)
                correct = (pred.argmax(1) == y_test).type(torch.float).sum().item()
                print("test correct ratio = ", correct / 100, "%")

    def fit(self, num_steps=1000):
        print("-------MP Optimizer---N = " + str(N) + "---alpha = " + str(alpha) + "-------")

        # 计算参数量
        para_num = torch.tensor(0).to(device)
        for para in self.net.parameters():
            para_num += para.reshape(-1).shape[0]
        proposal_nets = [1] * (self.N + 1)
        proposal_nets_paras = [1] * (self.N + 1)

        for s in tqdm(range(num_steps)):
            # 1. 建议参数
            proposal_nets[0] = copy.deepcopy(self.net).to(device)  # 初始化产生N个建议网络（可并行）
            proposal_nets_paras[0] = torch.tensor([]).to(device)

            for para in proposal_nets[0].parameters():
                proposal_nets_paras[0] = torch.cat((proposal_nets_paras[0], para.reshape(-1)))
            for i in range(1, self.N + 1):
                proposal_nets_paras[i] = torch.tensor([]).to(device)
                proposal_nets[i] = copy.deepcopy(self.update(proposal_nets[0])).to(device)
                for para in proposal_nets[i].parameters():
                    proposal_nets_paras[i] = torch.cat((proposal_nets_paras[i], para.reshape(-1)))

            self.step(s, proposal_nets, proposal_nets_paras, para_num)
        return np.array(self.loss_list)


init_network = Model()
init_network.load_state_dict(torch.load("CNN_model.pkl"))

# MP sample
network_MP = init_network.to(device)
trainer_MP = MPOptimizer(network_MP, alpha=alpha)
trainer_MP.N = N
samples_MP = trainer_MP.fit(num_steps=number_MP)
np.save('MP_alpha_' + str(alpha) + "_sample_number_" + str(number_MP) + "CNN_N_" + str(N) + '.npy', samples_MP)
