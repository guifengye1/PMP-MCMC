
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd

import math
# 正态分布概率密度函数计算
def normal(x, mu, sigma):
    numerator = np.exp((-(x - mu) ** 2) / (2 * sigma ** 2))
    denominator = sigma * np.sqrt(2 * np.pi)
    return numerator / denominator


def SP(hops, mu, sigma):
    states = []
    burn_in = int(hops * 0.2)
    current = random.uniform(-1 * sigma + mu, 1 * sigma + mu)
    acc_num = 0
    rej_num = 0
    for i in tqdm(range(hops)):
        # 记录样本
        states.append(current)
        # 1.建议
        movement = current + random.uniform(-0.25, 0.25)
        # 2.计算接受率
        curr_prob = normal(x=current, mu=mu, sigma=sigma)
        move_prob = normal(x=movement, mu=mu, sigma=sigma)
        acceptance = move_prob / (curr_prob + move_prob)
        # 3.判定是否接受
        event = random.uniform(0, 1)
        if acceptance > event:
            current = movement
            acc_num = acc_num + 1
        else:
            rej_num = rej_num + 1

    return states[burn_in:]


def MP(hops, mu, sigma, N):
    d = 1
    I = 0
    X = np.empty([hops * (N + 1), d])
    X[0, 0] = random.uniform(-1 * sigma + mu, 1 * sigma + mu)
    Y = np.empty([N + 1, d])
    Y[0, 0] = X[0, 0]
    # generate N new points from the proposal 初始化产生N个建议值（可并行）
    for i in range(N):
        j = i + 1
        Y[j, 0] = X[0, 0] + np.random.normal()
    K = np.empty([N + 1, d])
    A = np.empty([N + 1, d])
    for i in tqdm(range(hops)):
        # 计算接受率（可并行）
        for j in range(N + 1):
            temp = 1
            for k in range(N + 1):
                K[k] = stats.norm.pdf(Y[j, 0], Y[k, 0])
                if j != k:
                    temp = K[k] * temp
            A[j, 0] = temp * normal(x=Y[j, 0], mu=mu, sigma=sigma)

        # 根据接受率采样
        B = pd.DataFrame(A.reshape(-1))
        index = pd.DataFrame(np.linspace(0, N, N + 1).astype(np.int32))
        X[i * N + i:(i + 1) * (N + 1), 0] = Y[index.sample(N + 1, replace=True, weights=B[0]).values.reshape(-1), 0]
        # 重新产生N个建议值（可并行）
        I = np.random.choice(np.linspace(i * N + i, (i + 1) * (N + 1) - 1, N + 1).astype(np.int32), 1)
        Y[0, 0] = X[I, 0]
        for i in range(N):
            j = i + 1
            Y[j, 0] = X[I, 0] + np.random.normal()

    return X[int(0.2 * hops * (N + 1)):, 0]
def PSP(hops, mu, sigma,N):

    d = 1
    I = 0
    X = np.empty([hops * (N + 1), d])
    X[0, 0] = random.uniform(-1 * sigma + mu, 1 * sigma + mu)
    Y = np.empty([N + 1, d])
    tree_deep = math.log2(N + 1)

    Y[0, 0] = X[0, 0]
    for i in range(int(tree_deep)):
        j = int(math.pow(2, i))
        for k in range(int(j)):
            Y[k+j] = Y[k] + np.random.normal()

    A = np.ones([N + 1, d])
    weights = np.empty([N + 1, d])

    for i in tqdm(range(hops)):
        # 1.计算接受率（可并行）
        ## 1.1 计算似然函数
        for all in range(N + 1):
            weights[all, 0] = normal(Y[all, 0], mu=mu, sigma=sigma)
        ## 1.2 计算接收概率
        A = np.ones([N + 1, d])
        for all in range(N + 1):
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
                    w_new = weights[judg, 0] * stats.norm.pdf(Y[judg, 0], Y[judg + half_j, 0])
                    w_old = weights[judg + half_j, 0] * stats.norm.pdf(Y[judg + half_j, 0], Y[judg, 0])
                else:
                    w_new = weights[judg, 0] * stats.norm.pdf(Y[judg, 0], Y[judg - half_j, 0])
                    w_old = weights[judg - half_j, 0] * stats.norm.pdf(Y[judg - half_j, 0], Y[judg, 0])

                A[all, 0] = A[all, 0] * w_new / (w_new + w_old)

        B = pd.DataFrame(A.reshape(-1))
        index = pd.DataFrame(np.linspace(0, N, N + 1).astype(np.int32))
        X[i * N + i:(i + 1) * (N + 1), 0] = Y[index.sample(N + 1, replace=True, weights=B[0]).values.reshape(-1), 0]
        # 重新产生N个建议值（可并行）
        I = np.random.choice(np.linspace(i * N + i, (i + 1) * (N + 1) - 1, N + 1).astype(np.int32), 1)
        Y[0, 0] = X[I, 0]
        for i in range(int(tree_deep)):
            j = int(math.pow(2, i))
            for k in range(int(j)):
                Y[k + j] = Y[k] + np.random.normal()

    return X[int(0.2*hops*(N+1)):,0]


def PMP(hops, mu, sigma, N, deep):
    d = 1
    I = 0
    X = np.empty([hops * (N + 1) ** deep, d])
    X[0, 0] = random.uniform(-1 * sigma + mu, 1 * sigma + mu)
    Y = np.empty([(N + 1) ** deep, d])

    Y[0, 0] = X[0, 0]
    for dee in range(deep):
        temp = (N + 1) ** dee
        for j in range(N):
            for k in range(temp):
                Y[k + temp * (j + 1)] = Y[k] + np.random.normal()

    for hop in tqdm(range(hops)):
        A = np.ones([(N + 1) ** deep, d])
        weights = np.empty([(N + 1) ** deep, d])
        w_t = np.ones([N + 1, d])
        # 1.计算接受率（可并行）
        ## 1.1 计算似然函数
        for all in range((N + 1) ** deep):
            weights[all, 0] = normal(Y[all, 0], mu=mu, sigma=sigma)
        ## 1.2 计算接收概率
        for i in range(deep):
            temp = (N + 1) ** i
            for h in range((N + 1) ** i):
                for j in range(N + 1):
                    w_t[j, 0] = weights[h + j * temp, 0]
                for j in range(N + 1):
                    for k in range(N + 1):
                        if j != k:
                            w_t[j, 0] = w_t[j, 0] * stats.norm.pdf(Y[h + j * temp, 0], Y[h + k * temp, 0])
                for j in range(N + 1):
                    A[h + j * temp, 0] = A[h + j * temp, 0] * w_t[j, 0] / w_t.sum()
            if i < deep - 1:
                for l in range((N + 1) ** (i + 2) - (N + 1) ** (i + 1)):
                    A[l + (N + 1) ** (i + 1), 0] = A[(l + (N + 1) ** (i + 1)) % ((N + 1) * (i + 1)), 0]

        B = pd.DataFrame(A.reshape(-1))
        index = pd.DataFrame(np.linspace(0, (N + 1) ** deep - 1, (N + 1) ** deep).astype(np.int32))
        X[hop * (N + 1) ** deep:(hop + 1) * (N + 1) ** deep, 0] = Y[
            index.sample((N + 1) ** deep, replace=True, weights=B[0]).values.reshape(-1), 0]
        # 重新产生N个建议值（可并行）
        I = np.random.choice(
            np.linspace(hop * (N + 1) ** deep, (hop + 1) * (N + 1) ** deep - 1, (N + 1) ** deep).astype(np.int32), 1)

        Y[0, 0] = X[I, 0]
        for dee in range(deep):
            temp = (N + 1) ** dee
            for j in range(N):
                for k in range(temp):
                    Y[k + temp * (j + 1)] = Y[k] + np.random.normal()

    return X[int(0.2 * hops * (N + 1) ** deep):, 0]
mu, sigma = 0,1
times = 20
number_sample = 10

sp = np.empty(times)
mp = np.empty(times)
pmp = np.empty(times)
for i in range(times):
    sp[i] = np.array(SP(number_sample,mu=mu,sigma=sigma)).mean()
    mp[i] = np.array(MP(number_sample, mu=mu, sigma=sigma, N=7)).mean()
    pmp[i] = np.array(PMP(number_sample, mu=mu, sigma=sigma,N=7,deep=2)).mean()
print(sp)
print(mp)
print(pmp)

data = np.ones([times,3])
data[:,0]=sp
data[:,1]=mp
data[:,2]=pmp
fig = plt.figure(figsize=(3,10))
plt.title("ERROR")
labels = 'SP','MP','PMP'
view = plt.boxplot(data,labels=labels)
plt.savefig("error.pdf")