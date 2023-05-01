import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import math

# 正态分布概率密度函数计算
def normal(data,mu,cov):
    rv = multivariate_normal(mu, cov)
    return rv.pdf(data)

# 定义转移
def transition_prob(x, y,d):
    # 计算从 x 转移到 y 的概率
    # 在这里，我们假设建议函数是对称的，即 p(y|x) = p(x|y)
    return 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * np.sum((y - x)**2) / sigma**2) *10**(d/10)


def PMP(hops, mu, cov, N, dim):
    d = dim
    I = 0
    X = np.empty([hops * (N + 1), d])
    X[0, :] = np.ones([1, dim]) * 2.5
    Y = np.empty([N + 1, d])
    Y[0, :] = X[0, :]

    tree_deep = math.log2(N + 1)

    for i in range(int(tree_deep)):
        j = int(math.pow(2, i))
        for k in range(int(j)):
            Y[k + j, :] = np.random.normal(Y[k, :].reshape(-1), sigma, size=Y[k, :].shape[0]).reshape(Y[k, :].shape[0])

    for i in range(hops):

        A = np.ones([(N + 1), 1])
        weights = np.empty([(N + 1), 1])
        w_t = np.ones([N + 1, 1])
        # 1.计算接受率（可并行）
        ## 1.1 计算似然函数
        for all in range(N + 1):
            weights[all, 0] = normal(Y[all, :], mu=mu, cov=cov)

        ## 1.2 计算接收概率
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
                    w_new = weights[judg, 0] * transition_prob(Y[judg, :], Y[judg + half_j, :],d)
                    w_old = weights[judg + half_j, 0] * transition_prob(Y[judg + half_j, :], Y[judg, :],d)
                else:
                    w_new = weights[judg, 0] * transition_prob(Y[judg, :], Y[judg - half_j, :],d)
                    w_old = weights[judg - half_j, 0] * transition_prob(Y[judg - half_j, :], Y[judg, :],d)
                # print( w_new / (w_new + w_old))
                A[all, 0] = A[all, 0] * w_new / (w_new + w_old)

        B = pd.DataFrame(A.reshape(-1))

        index = pd.DataFrame(np.linspace(0, N, N + 1).astype(np.int32))
        X[i * N + i:(i + 1) * (N + 1), :] = Y[index.sample(N + 1, replace=True, weights=B[0]).values.reshape(-1), :]

        # 重新产生N个建议值（可并行）
        I = np.random.choice(np.linspace(i * N + i, (i + 1) * (N + 1) - 1, N + 1).astype(np.int32), 1)

        Y[0, :] = X[I, :]

        for i in range(int(tree_deep)):
            j = int(math.pow(2, i))
            for k in range(int(j)):
                Y[k + j, :] = np.random.normal(Y[k, :].reshape(-1), sigma, size=Y[k, :].shape[0]).reshape(
                    Y[k, :].shape[0])

    return X[:, :]
nums=[50,100,150,200,250]
Ns = [1,3,7,15,31]
dims = [10,20,40,80,160]
print("维度","\t深度","\t链长","\t\t均值","\t\t方差")
c1=[]
c2=[]
c3=[]
c4=[]
c5=[]
for dim in dims:
    for N in Ns:

        # 设置均值向量和协方差矩阵
        mu = np.zeros(dim)
        cov = np.eye(dim)
        sigma= 0.5
        # 采样获得样本
        dist = PMP(500,mu=mu,cov=cov,N=N,dim=dim)
        dist = np.array(dist)
        for num in nums:
            temp = dist[:num*(N+1),:] # 计算样本偏度值
            print(dim, "\t", math.log2(N + 1), "\t",num,"\t",temp.mean(), "\t\t",temp.std())
            c1.append(dim)
            c2.append(math.log2(N + 1))
            c3.append(num)
            c4.append(temp.mean())
            c5.append(temp.std())
        print("---------------------------------------------")
df= pd.DataFrame({'维度':c1,'深度':c2,'链长':c3,'均值':c4,"标准差":c5})
df.to_csv("dimension_Chins_Parl.csv",index=False)