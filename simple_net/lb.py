import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.distributions as dist
import copy
import random
import pandas as pd


N = 100000
beta0 = torch.tensor([-1.0])
beta_true = torch.tensor([2.0])
sigma_true = torch.tensor([0.5])

X = dist.Uniform(-1,1).sample((N,))
Y = beta0 + X *beta_true + sigma_true*torch.normal(0,1, (N,))
data = {'x' : X,'y' : Y}

class BayesNet_o(nn.Module):
    def __init__(self, seed = 42):
        super().__init__()
        torch.random.manual_seed(seed)
        # Initialize the parameters with some random values
        self.beta0 = nn.Parameter(torch.tensor([0.0]))
        self.beta = nn.Parameter(torch.tensor([0.0]))
        self.sigma = nn.Parameter(torch.tensor([1.0]) ) # this has to be positive
    def forward(self, data: dict):
        return self.beta0 + self.beta*data['x']

    def loglik(self, data):
        # Evaluates the log of the likelihood given a set of X and Y
        yhat = self.forward(data)
        logprob = dist.Normal(yhat, self.sigma.abs()).log_prob(data['y'])
        return logprob.sum() / data["y"].shape[0] * 50

    def logprior(self):
        # Return a scalar of 0 since we have uniform prior
        return torch.tensor(0.0)

    def logpost(self, data):
        return self.loglik(data) + self.logprior()




class MetropolisOptimizer():
    def __init__(self, net, alpha):
        super().__init__()
        self.net = net
        self.alpha = alpha
        self.d = sum(p.numel() for p in self.net.parameters())

    @torch.no_grad()
    def step(self, data=None):
        # Step 1:
        proposal_net = copy.deepcopy(
            self.net)  # we copy the whole network instead of just the parameters for simplicity
        for name, par in proposal_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)

        # Step 2: calculate ratio

        ratio = torch.exp(proposal_net.logpost(data) - self.net.logpost(data))

        # print(proposal_net.logpost(data) - self.net.logpost(data),ratio)
        # Step 3: update with some probability:
        if (random.random() < ratio).bool():
            self.net = proposal_net
        else:
            pass
        return self.net

    def fit(self, data=None, num_steps=1000):
        # We create one tensor per parameter so that we can keep track of the parameter values over time:
        parameter_trace = np.empty([num_steps, self.d])

        for s in tqdm(range(num_steps)):
            current_net = self.step(data)
            j = 0
            for key, val in current_net.named_parameters():
                parameter_trace[s:(s + 1), j] = val.data
                j = j + 1
        return parameter_trace


from scipy import stats


class BayesNet(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        torch.random.manual_seed(seed)
        # Initialize the parameters with some random values
        self.beta0 = nn.Parameter(torch.tensor([0.0]))
        self.beta = nn.Parameter(torch.tensor([0.0]))
        self.sigma = nn.Parameter(torch.tensor([1.0]))  # this has to be positive

    def forward(self, data: dict):
        return self.beta0 + self.beta * data['x']

    def loglik(self, data):
        # Evaluates the log of the likelihood given a set of X and Y
        yhat = self.forward(data)
        logprob = dist.Normal(yhat, self.sigma.abs()).log_prob(data['y'])

        return logprob.sum() / data["y"].shape[0] * 50


def log_trans_prob(net, net_star):
    log_trans_prob = torch.tensor([0], dtype=torch.float64)
    for p, p_star in zip(net.parameters(), net_star.parameters()):
        log_trans_prob = log_trans_prob + torch.log(
            torch.tensor([stats.norm.pdf(p.detach().numpy()[0], p_star.detach().numpy()[0])]))
    return log_trans_prob





class GMOptimizer():
    def __init__(self, net, alpha, N):
        super().__init__()
        self.net = net  # model
        self.alpha = alpha  # learning radio
        self.N = N  # parallel number
        self.d = sum(p.numel() for p in self.net.parameters())  # 参数维度为d

    @torch.no_grad()
    def update(self,net):
        my_net = copy.deepcopy(net)
        for name, par in my_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)
        return my_net

    @torch.no_grad()
    def step(self, data=None, proposal_nets=None):
        # Step 1:
        # 计算接受率（可并行）
        K = np.empty([self.N + 1, 1])
        A = np.empty([self.N + 1, 1])
        for j in range(self.N + 1):
            temp = 0
            for k in range(self.N + 1):
                K[k] = log_trans_prob(proposal_nets[j], proposal_nets[k]).item()
                if j != k:
                    temp += K[k]
            A[j, 0] = temp + proposal_nets[j].loglik(data).item()
            # print("proposal_nets[j].loglik(data).item()",proposal_nets[j].loglik(data).item())
        # 根据接受率采样

        B = pd.DataFrame(np.exp(A).reshape(-1))
        index = pd.DataFrame(np.linspace(0, self.N, self.N + 1).astype(np.int32))
        index_weight = index.sample(self.N + 1, replace=True, weights=B[0]).values.reshape(-1)

        new_proposal_nets = copy.deepcopy(proposal_nets)
        for i, j in zip(index_weight, index.values.reshape(-1)):
            new_proposal_nets[j] = proposal_nets[i]

        I = np.random.choice(np.linspace(0, self.N, self.N + 1).astype(np.int32), 1)
        self.net = new_proposal_nets[I[0]]
        return new_proposal_nets

    def fit(self, data=None, num_steps=1000):
        # We create one tensor per parameter so that we can keep track of the parameter values over time:

        parameter_trace = np.empty([num_steps*(self.N+1), self.d])
        proposal_nets = {}
        for s in tqdm(range(num_steps)):
            # 1.建议参数
            proposal_nets[0] = copy.deepcopy(self.net)
            # generate N new points from the proposal 初始化产生N个建议网络（可并行）
            for i in range(1, self.N + 1):
                proposal_nets[i] = copy.deepcopy(self.update(proposal_nets[0]))
            # 2.优化采样
            current_nets = self.step(data, proposal_nets)

            # 3.记录样本
            j = 0
            for i,current_net in zip(range(self.N+1), current_nets):
                for key, val in current_nets[current_net].named_parameters():
                    parameter_trace[s*(self.N+1)+i:(s*(self.N+1)+i+1), j] = val.data
                    j = (j + 1)% 3
        return parameter_trace


class preMOptimizer():
    def __init__(self, net, alpha, N):
        super().__init__()
        self.net = net  # model
        self.alpha = alpha  # learning radio
        self.N = N  # parallel number log2(N+1)
        self.d = sum(p.numel() for p in self.net.parameters())  # 参数维度为d

    @torch.no_grad()
    def update(self,net):
        my_net = copy.deepcopy(net)
        for name, par in my_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)
        return my_net

    @torch.no_grad()
    def step(self, data=None, proposal_nets=None):
        # Step 1:
        # 计算接受率（可并行）
        tree_deep = math.log2(self.N + 1)
        K = np.empty([self.N + 1, 1])
        A = np.zeros([self.N + 1, 1])
        weights = np.empty([self.N + 1, 1])
        for all in range(self.N + 1):
            weights[all, 0] = np.exp(proposal_nets[all].loglik(data).item())
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
                    w_new = weights[judg, 0] * np.exp(
                        log_trans_prob(proposal_nets[judg], proposal_nets[judg + half_j]).item())
                    w_old = weights[judg + half_j, 0] * np.exp(
                        log_trans_prob(proposal_nets[judg + half_j], proposal_nets[judg]).item())

                else:
                    w_new = weights[judg, 0] * np.exp(
                        log_trans_prob(proposal_nets[judg], proposal_nets[judg - half_j]).item())
                    w_old = weights[judg - half_j, 0] * np.exp(
                        log_trans_prob(proposal_nets[judg - half_j], proposal_nets[judg]).item())

                A[all, 0] = A[all, 0] + np.log(w_new / (w_new + w_old))

        # A = (A - np.mean(A)) / np.std(A)
        # 根据接受率采样
        B = pd.DataFrame(np.exp(A).reshape(-1))

        index = pd.DataFrame(np.linspace(0, self.N, self.N + 1).astype(np.int32))

        index_weight = index.sample(self.N + 1, replace=True, weights=B[0]).values.reshape(-1)

        new_proposal_nets = copy.deepcopy(proposal_nets)
        for i, j in zip(index_weight, index.values.reshape(-1)):
            new_proposal_nets[j] = proposal_nets[i]

        I = np.random.choice(np.linspace(0, self.N, self.N + 1).astype(np.int32), 1)

        self.net = new_proposal_nets[I[0]]

        return new_proposal_nets

    def fit(self, data=None, num_steps=1000):
        # We create one tensor per parameter so that we can keep track of the parameter values over time:

        parameter_trace = np.empty([num_steps, self.d])
        proposal_nets = {}
        tree_deep = math.log2(self.N + 1)

        for s in tqdm(range(num_steps)):
            proposal_nets[0] = copy.deepcopy(self.net)
            for i in range(int(tree_deep)):
                j = int(math.pow(2, i))
                for k in range(int(j)):
                    proposal_nets[k + j] = copy.deepcopy(self.update(proposal_nets[k]))

            current_nets = self.step(data, proposal_nets)
            j = 0
            for key, val in self.net.named_parameters():
                parameter_trace[s:(s + 1), j] = val.data
                j = j + 1
        return parameter_trace


from tqdm import tqdm



class GMpreOptimizerV2():
    def __init__(self, net, alpha, N, deep):
        super().__init__()
        self.net = net  # model
        self.alpha = alpha  # learning radio
        self.N = N  # parallel number
        self.deep = deep
        self.d = sum(p.numel() for p in self.net.parameters())  # 参数维度为d

    @torch.no_grad()
    def update(self,net):
        my_net = copy.deepcopy(net)
        for name, par in my_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)
        return my_net

    @torch.no_grad()
    def step(self, data=None, proposal_nets=None):
        # Step 1:
        # 计算接受率（可并行）

        A = np.ones([(self.N + 1) ** self.deep, 1])
        weights = np.empty([(self.N + 1) ** self.deep, 1])
        w_t = np.ones([self.N + 1, 1])
        for all in range((self.N + 1) ** self.deep):
            weights[all, 0] = np.exp(proposal_nets[all].loglik(data).item())

        # 计算接受率
        for i in range(self.deep):
            temp = (self.N + 1) ** i
            for h in range((self.N + 1) ** i):

                for j in range(self.N + 1):
                    w_t[j, 0] = weights[h + j * temp, 0]
                for j in range(self.N + 1):
                    for k in range(self.N + 1):
                        if j != k:
                            w_t[j, 0] = w_t[j, 0] * np.exp(
                                log_trans_prob(proposal_nets[h + j * temp], proposal_nets[h + k * temp]).item())
                for j in range(self.N + 1):
                    A[h + j * temp, 0] = A[h + j * temp, 0] * w_t[j, 0] / w_t.sum()
            if i < self.deep - 1:
                for l in range((self.N + 1) ** (i + 2) - (self.N + 1) ** (i + 1)):
                    A[l + (self.N + 1) ** (i + 1), 0] = A[(l + (self.N + 1) ** (i + 1)) % ((self.N + 1) * (i + 1)), 0]

        # 根据接受率采样

        B = pd.DataFrame(A.reshape(-1))
        index = pd.DataFrame(np.linspace(0, (self.N + 1) ** self.deep - 1, (self.N + 1) ** self.deep).astype(np.int32))
        index_weight = index.sample((self.N + 1) ** self.deep, replace=True, weights=B[0]).values.reshape(-1)

        new_proposal_nets = copy.deepcopy(proposal_nets)
        for i, j in zip(index_weight, index.values.reshape(-1)):
            new_proposal_nets[j] = proposal_nets[i]

        I = np.random.choice(np.linspace(0, (self.N + 1) ** self.deep - 1, (self.N + 1) ** self.deep).astype(np.int32),
                             1)
        self.net = new_proposal_nets[I[0]]
        return new_proposal_nets

    def fit(self, data=None, num_steps=1000):
        # We create one tensor per parameter so that we can keep track of the parameter values over time:

        parameter_trace = np.empty([num_steps*(self.N+1)**self.deep, self.d])
        proposal_nets = {}
        for s in tqdm(range(num_steps)):
            # 1.建议参数
            proposal_nets[0] = copy.deepcopy(self.net)
            # generate N new points from the proposal 初始化产生N个建议网络（可并行）
            for i in range(self.deep):
                temp = (self.N + 1) ** i
                for j in range(self.N):
                    for k in range(temp):
                        proposal_nets[k + temp * (j + 1)] = copy.deepcopy(self.update(proposal_nets[k]))
            # 2.优化采样
            current_nets = self.step(data, proposal_nets)
            # 3.记录样本
            j = 0
            for i,current_net in zip(range((self.N+1)**self.deep),current_nets):
                for key, val in current_nets[current_net].named_parameters():
                    parameter_trace[s*(self.N+1)**self.deep+i:(s*(self.N+1)**self.deep+i+1), j] = val.data
                    j = (j + 1)%3
        return parameter_trace


import matplotlib.pyplot as plt
import numpy as np
import math


steps = [0.05, 0.1, 0.2, 0.4]
number_opt = 3
fig = plt.figure(figsize=(13, 4))
for i, step in zip(range(len(steps)), steps):
    # 1.MH
    model = BayesNet_o()
    # 实例化优化器，优化的对象就是上述模型
    sample_num = 1000
    trainer = MetropolisOptimizer(model, alpha=step)
    parameter_trace = trainer.fit(data, num_steps=sample_num+1500)

    line = np.linspace(0, sample_num, sample_num)
    fig_1 = plt.subplot(number_opt, len(steps), i + 1)
    plt.ylim(-1.5, -0.5)
    plt.title("step = "+str(step))
    plt.plot(line, parameter_trace[parameter_trace.shape[0] - sample_num:parameter_trace.shape[0], 0])

    # 2.GMP
    model = BayesNet()
    # 实例化优化器，优化的对象就是上述模型
    trainer = GMOptimizer(model, alpha=step, N=7)
    sample_num = sample_num * 8
    parameter_trace = trainer.fit(data, num_steps=int((sample_num+1500)/8))

    line = np.linspace(0, sample_num, sample_num)
    fig_2 = plt.subplot(number_opt, len(steps), len(steps) + i + 1)
    plt.ylim(-1.5, -0.5)
    plt.plot(line, parameter_trace[parameter_trace.shape[0] - sample_num:parameter_trace.shape[0], 0])

    # 3. PMP
    model = BayesNet()
    # 实例化优化器，优化的对象就是上述模型
    trainer = GMpreOptimizerV2(model, alpha=step, N=7, deep=2)
    sample_num = sample_num * 8
    parameter_trace = trainer.fit(data, num_steps=int((sample_num+2000)/64))

    line = np.linspace(0, sample_num, sample_num)
    fig_3 = plt.subplot(number_opt, len(steps), 2*len(steps) + i + 1)
    plt.ylim(-1.5, -0.5)
    plt.plot(line, parameter_trace[parameter_trace.shape[0] - sample_num:parameter_trace.shape[0], 0])


plt.subplots_adjust(hspace=0.5, wspace=0.5)



plt.savefig("lb.pdf")
