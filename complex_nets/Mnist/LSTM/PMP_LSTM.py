import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
from time import time
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
alpha = 0.0001
num_steps = 1000000
N = 7
# 定义超参数
batch_size = 60000 # 一次训练的样本数目

# 1. 模型定义
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # LSTM层
        self.rnn = nn.LSTM(
            input_size=28,  # 图片每行的像素点
            hidden_size=64,  # 隐藏层神经元个数
            num_layers=1,  # RNN layers数量
            batch_first=True  # 以批量为第一维度：(batch, time_step, input_size)
        )
        # 全连接层
        self.out = nn.Linear(64, 10)  # 输出特征维度为10,：0-9的数字分类

    def forward(self, x):
        output, (h_n, h_c) = self.rnn(x, None)  # # None 表示 hidden state 会用全0的 state
        # output维度：(batch, time_step, output_size),time_step表示rnn的时间步数或者图片高度
        # h_n维度：(n_layers, batch, hidden_size), LSTM的分线
        # h_c维度：(n_layers, batch, hidden_size), LSTM的主线

        # 由于输出为时间序列，因而最终目标应该是最后时间点time_step的输出值
        output = self.out(output[:, -1, :])
        return output



@torch.no_grad()
def loss(net):
    yhat = net(X)
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(yhat, y) / 10


def get_data():
    """获取数据"""

    # 获取测试集
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
    train_loader = DataLoader(train, batch_size=batch_size)  # 分割测试集

    # 获取测试集
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    # 返回分割好的训练集和测试集
    return train_loader, test_loader
train_loader, test_loader = get_data()
for _, (X, y) in enumerate(train_loader):
    X = X.view(-1,28,28).to(device)
    y = y.to(device)
for _, (x_test,y_test) in enumerate(test_loader):
    x_test = x_test.view(-1,28,28).to(device)
    y_test = y_test.to(device)


# 定义优化器
class PMPOptimizer():
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
    def update(self, net):
        my_net = copy.deepcopy(net)
        for name, par in my_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)
        return my_net


    @torch.no_grad()
    def step(self, s, proposal_nets=None, proposal_nets_paras=None, para_num=None):
        # Step 1:
        tree_deep = int(math.log2(self.N + 1))
        # 计算接受率（可并行）
        A = torch.zeros([self.N + 1, 1]).to(device)
        weights = torch.empty([self.N + 1, 1]).to(device)
        tran = torch.zeros([self.N + 1, self.N + 1]).to(device)
        for i in range(tree_deep):
            for j in range(2 ** i):
                tran[j][j + 2**i] = (self.const_1 - (proposal_nets_paras[j] - proposal_nets_paras[j + 2**i]) ** 2 * self.const_2).sum() / para_num
                tran[j + 2**i][j] = tran[j][j + 2**i]
        for all in range(self.N + 1):
            weights[all, 0] = torch.exp(-loss(proposal_nets[all]))
        for all in range(self.N + 1):
            for c in range(int(tree_deep)):
                judg = all
                j = int(math.pow(2, c + 1))
                half_j = int(j / 2)
                if judg > 0:
                    deep = int(math.log2(judg))
                while (judg > j - 1):
                    if (judg >= int(2 ** deep)):
                        judg = judg - int(2 ** deep)
                    deep = deep - 1
                if (judg < half_j):
                    w_new = weights[judg, 0] * tran[judg][judg+half_j]
                    w_old = weights[judg + half_j, 0] * tran[judg+half_j][judg]
                else:
                    w_new = weights[judg, 0] * tran[judg][judg - half_j]
                    w_old = weights[judg - half_j, 0] * tran[judg - half_j][judg]
                A[all, 0] = A[all, 0] + torch.log(w_new / (w_new + w_old))

        mean_a = torch.mean(A)
        std_a = torch.std(A)
        A = (A - mean_a) / std_a
        B = torch.exp(A).reshape(-1)
        I = torch.multinomial(B, 1, replacement=True).data
        self.net = proposal_nets[I.cpu().item()]
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

        return self.net

    def fit(self, num_steps=1000):
        # 计算参数量
        para_num = torch.tensor(0).to(device)
        for para in self.net.parameters():
            para_num += para.reshape(-1).shape[0]

        proposal_nets = [1] * (self.N +1)
        proposal_nets_paras = [1] * (self.N + 1)
        tree_deep = math.log2(self.N + 1)

        for s in tqdm(range(num_steps)):
            # 1.建议参数
            proposal_nets[0] = copy.deepcopy(self.net).to(device)# 初始化产生N个建议网络（可并行）
            proposal_nets_paras[0] = torch.tensor([]).to(device)
            for para in proposal_nets[0].parameters():
                proposal_nets_paras[0] = torch.cat((proposal_nets_paras[0], para.reshape(-1)))

            for i in range(int(tree_deep)):
                j = int(math.pow(2, i))
                for k in range(int(j)):
                    proposal_nets_paras[k + j] = torch.tensor([]).to(device)
                    proposal_nets[k + j] = copy.deepcopy(self.update(proposal_nets[k])).to(device)
                    for para in proposal_nets[k+j].parameters():
                        proposal_nets_paras[k+j] = torch.cat((proposal_nets_paras[k+j], para.reshape(-1)))

            # 2.优化采样
            self.step(s, proposal_nets, proposal_nets_paras, para_num)
        return np.array(self.loss_list)

init_network = Model()
init_network.load_state_dict(torch.load("LSTM_model.pkl"))

# MPM sample
init_network = init_network.to(device)
trainer_PMP = PMPOptimizer(init_network, alpha=alpha)
trainer_PMP.N = N
samples_PMP = trainer_PMP.fit(num_steps=num_steps)

np.save('PMP_alpha_'+str(alpha)+"_sample_number_"+str(num_steps)+"LSTM_N_"+str(N)+'.npy', samples_PMP)




