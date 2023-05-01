import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import  nn
import numpy as np
from tqdm import tqdm
import random,copy
from time import time


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
number_MH = 250000
alpha = 0.0001
# 定义超参数
batch_size = 60000  # 一次训练的样本数

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
    X = X.to(device)
    y = y.to(device)
for _, (x_test,y_test) in enumerate(test_loader):
    x_test = x_test.to(device)
    y_test = y_test.to(device)


def loss(net):
    with torch.no_grad():
        yhat = net(X)
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(yhat, y)

class MetropolisOptimizer():
    def __init__(self, net, alpha):
        super().__init__()
        self.net = net
        self.alpha = alpha
        self.first = 0
        self.loss = None
        self.loss_proposal = None
        self.lamb = 10000
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_list = []
    @torch.no_grad()
    def update(self, net):
        my_net = copy.deepcopy(net)
        for name, par in my_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)
        return my_net

    @torch.no_grad()
    def step(self,s):

        ## 整个 step 用时 0.09s
        # Step 1:
        proposal_net = copy.deepcopy(self.update(self.net)).to(device)
        self.loss_proposal = loss(proposal_net)
        ratio = torch.exp(self.lamb * (self.loss -self.loss_proposal))

        # Step 3: update with some probability:

        if torch.rand(1).to(device) < ratio:
            self.net = proposal_net
            self.loss = self.loss_proposal
        else:

            pass
        self.loss_list.append(self.loss)

        if s % 1000 == 0:
            print( "loss = ", self.loss)
            if s % 10000 ==0:
                pred = self.net(X)
                correct = (pred.argmax(1) == y).type(torch.float).sum().item()
                print("train correct ratio = ",correct / batch_size * 100,"%")


        return self.net

    def fit(self, num_steps=1000):
        # # We create one tensor per parameter so that we can keep track of the parameter values over time:
        # self.parameter_trace = {
        #     key: torch.zeros((num_steps,) + par.size()) for key, par in self.net.named_parameters()}

        self.loss = loss(self.net)
        for s in tqdm(range(num_steps)):
            current_net = self.step(s)
            if s % 10000 == 0:
                pred = self.net(x_test)

                correct = (pred.argmax(1) == y_test).type(torch.float).sum().item()
                print("\n test correct ratio = ", correct / 100, "%")
        return np.array(self.loss_list)

init_network = Model()
init_network.load_state_dict(torch.load("CNN_model.pkl"))
network_MH = init_network.to(device)

trainer_MH = MetropolisOptimizer(network_MH, alpha=alpha)

samples_MH = trainer_MH.fit(num_steps=number_MH)
np.save('MH_alpha_'+str(alpha)+"_sample_number_"+str(number_MH)+"CNN"+'.npy', samples_MH)



