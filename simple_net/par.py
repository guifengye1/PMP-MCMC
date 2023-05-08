import torch
import torch.nn as nn
import torch.distributions as dist
from torch.optim.optimizer import Optimizer
import copy
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm import tqdm
import numpy as np
import math

# generate data
N = 50
beta0 = torch.tensor([-1.0])
beta_true = torch.tensor([2.0])
sigma_true = torch.tensor([0.5])
X = dist.Uniform(-1,1).sample((N,))
Y = beta0 + X *beta_true + sigma_true*torch.normal(0,1, (N,))
data = {'x' : X,'y' : Y}

# network for MetropolisOptimizer
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

# network for GMOptimizer(MP_MCMC) and GMpreOptimizerV2(PMP_MCMC)
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

class GMOptimizer():
    def __init__(self, net, alpha, N):
        super().__init__()
        self.net = net  # model
        self.alpha = alpha  # learning radio
        self.N = N  # parallel number
        self.d = sum(p.numel() for p in self.net.parameters())  # The parameter dimension is d

    @torch.no_grad()
    def update(self, net):
        my_net = copy.deepcopy(net)
        for name, par in my_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)
        return my_net

    @torch.no_grad()
    def step(self, data=None, proposal_nets=None):
        # Step 1:
        # Calculate acceptance rate (parallelizable)
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
        # Sampling by acceptance rate
        B = pd.DataFrame(np.exp(A).reshape(-1))
        index = pd.DataFrame(np.linspace(0, self.N, self.N + 1).astype(np.int32))
        index_weight = index.sample(self.N + 1, replace=True, weights=B[0]).values.reshape(-1)
        new_proposal_nets = copy.deepcopy(proposal_nets)
        for i, j in zip(index_weight, index.values.reshape(-1)):
            new_proposal_nets[j] = proposal_nets[i]
        # Randomly choose one of them as the initial value for the next sampling
        I = np.random.choice(np.linspace(0, self.N, self.N + 1).astype(np.int32), 1)
        self.net = new_proposal_nets[I[0]]
        return new_proposal_nets

    def fit(self, data=None, num_steps=1000):
        parameter_trace = np.empty([num_steps, self.d])
        proposal_nets = {}
        for s in tqdm(range(num_steps)):
            # 1.Generate N propose network 
            proposal_nets[0] = copy.deepcopy(self.net)
            for i in range(1, self.N + 1):
                proposal_nets[i] = copy.deepcopy(self.update(proposal_nets[0]))

            # 2.sampling
            current_nets = self.step(data, proposal_nets)
            # 3.record sample
            j = 0
            for key, val in current_nets[0].named_parameters():
                parameter_trace[s:(s + 1), j] = val.data
                j = j + 1
        return parameter_trace

class GMpreOptimizerV2():
    def __init__(self, net, alpha, N, deep):
        super().__init__()
        self.net = net  # model
        self.alpha = alpha  # learning radio
        self.N = N  # parallel number
        self.deep = deep
        self.d = sum(p.numel() for p in self.net.parameters())  # The parameter dimension is d

    @torch.no_grad()
    def update(self, net):
        my_net = copy.deepcopy(net)
        for name, par in my_net.named_parameters():
            newpar = par + torch.normal(torch.zeros_like(par), self.alpha)
            par.copy_(newpar)
        return my_net

    @torch.no_grad()
    def step(self, data=None, proposal_nets=None):
        A = np.ones([(self.N + 1) ** self.deep, 1])
        weights = np.empty([(self.N + 1) ** self.deep, 1])
        w_t = np.ones([self.N + 1, 1])
        # Step 1:
        # Calculate the likelihood (parallelizable)
        for all in range((self.N + 1) ** self.deep):
            weights[all, 0] = np.exp(proposal_nets[all].loglik(data).item())

        # Calculate acceptance rate
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

        # Sampling by acceptance rate
        B = pd.DataFrame(A.reshape(-1))
        index = pd.DataFrame(np.linspace(0, (self.N + 1) ** self.deep - 1, (self.N + 1) ** self.deep).astype(np.int32))
        index_weight = index.sample((self.N + 1) ** self.deep, replace=True, weights=B[0]).values.reshape(-1)

        new_proposal_nets = copy.deepcopy(proposal_nets)
        for i, j in zip(index_weight, index.values.reshape(-1)):
            new_proposal_nets[j] = proposal_nets[i]
        # Randomly choose one of them as the initial value for the next sampling
        I = np.random.choice(np.linspace(0, (self.N + 1) ** self.deep - 1, (self.N + 1) ** self.deep).astype(np.int32),
                             1)
        self.net = new_proposal_nets[I[0]]
        return new_proposal_nets

    def fit(self, data=None, num_steps=1000):
        parameter_trace = np.empty([num_steps, self.d])
        proposal_nets = {}
        for s in tqdm(range(num_steps)):
            # 1.Generate (self.N + 1) ** self.deep propose network 
            proposal_nets[0] = copy.deepcopy(self.net)
            for i in range(self.deep): # (parallelizable)
                temp = (self.N + 1) ** i
                for j in range(self.N):
                    for k in range(temp):
                        proposal_nets[k + temp * (j + 1)] = copy.deepcopy(self.update(proposal_nets[k]))
            # 2.sampling
            current_nets = self.step(data, proposal_nets)
            # 3.record sample
            j = 0
            for key, val in current_nets[0].named_parameters():
                parameter_trace[s:(s + 1), j] = val.data
                j = j + 1
        return parameter_trace


# 1.MH
model = BayesNet_o()
# Instantiate the optimizer, the optimized object is the above model
trainer = MetropolisOptimizer(model, alpha=0.02)
parameter_trace = trainer.fit(data, num_steps=1000)
sample_num = parameter_trace.shape[0]
line = np.linspace(0, sample_num, sample_num)
colors = 'bgr'
for i, j in zip(range(trainer.d), colors):
    plt.plot(line, parameter_trace[:, i], label="SP par " + str(i), color=j, linestyle="-")

# 2.GMP
model = BayesNet()
# Instantiate the optimizer, the optimized object is the above model
trainer = GMOptimizer(model, alpha=0.02, N=7)
parameter_trace = trainer.fit(data, num_steps=1000)
sample_num = parameter_trace.shape[0]
line = np.linspace(0, sample_num, sample_num)
for i, j in zip(range(trainer.d), colors):
    plt.plot(line, parameter_trace[:, i], label="MP par " + str(i), color=j, linestyle=":")

model = BayesNet()
# Instantiate the optimizer, the optimized object is the above model
trainer = GMpreOptimizerV2(model, alpha=0.02, N=7, deep=3)
parameter_trace = trainer.fit(data, num_steps=1000)
sample_num = parameter_trace.shape[0]
line = np.linspace(0, sample_num, sample_num)
for i, j in zip(range(trainer.d), colors):
    plt.plot(line, parameter_trace[:, i], label="preMP par " + str(i), color=j, linestyle="-.")

plt.legend()
plt.savefig("par.pdf")