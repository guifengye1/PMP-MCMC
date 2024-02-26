import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import torchbnn as bnn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import argparse  #导入argparse模块
parser = argparse.ArgumentParser(description='argparse learning')  # 创建解析器
parser.add_argument('--gpu', type=str, default=0, help='input an integer')  # 添加参数
parser.add_argument('--N', type=int, default=0, help='input an integer')  # 添加参数

args = parser.parse_args()  # 解析参数
device = 'cuda:'+args.gpu
N = 2**(args.N)-1



batch_size = 60000
device = device if torch.cuda.is_available() else 'cpu'  # 0.001
print("device :",device,"\nparllel:",N)
import copy
from tqdm import tqdm

# get data
def get_data():
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(), # to tensor
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # standardization
                                       ]))
    train_loader = DataLoader(train, batch_size=batch_size)  
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # to tensor
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))  # standardization
                                      ]))
    test_loader = DataLoader(test, batch_size=batch_size)  
    return train_loader, test_loader
train_loader, test_loader = get_data()
for _, (X, y) in enumerate(train_loader):
    X = X.reshape(batch_size,-1).to(device)
    y = y.to(device)
for _, (x_test,y_test) in enumerate(test_loader):
    x_test = x_test.reshape(10000,-1).to(device)
    y_test = y_test.to(device)

class bnnPMPHmc():
    def __init__(self,net,alpha,N):
        super().__init__()
        self.net = net
        self.alpha = alpha
        self.N = N
        self.d = sum(p.numel() for p in self.net.parameters())
        self.loss = torch.nn.CrossEntropyLoss().to(device)
        self.loss_list = []
        self.train_acc = []
        self.test_acc = []

    def step(self, s, p_s, nets_loss,tree_deep):
        A = torch.ones([self.N + 1, 1]).to(device)
        t_1 = torch.FloatTensor([1]).to(device)
        t_0 = torch.FloatTensor([0]).to(device)
        # 1.Calculate acceptance ratio
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
                    w_old = torch.min(t_1,w_old/w_new)
                    w_new = torch.max(t_0,1-w_old/w_new)
                    #print(judg+half_j,"to",judg,"  new:",ww_new,"  old:",ww_old,"  w_new:",nets_loss[judg],"  w_old:",nets_loss[judg+half_j])
                else:
                    w_new = torch.exp(nets_loss[judg] - (p_s[judg][judg - half_j]*p_s[judg][judg - half_j]).sum()/2)
                    w_old =torch.exp(nets_loss[judg-half_j] - (p_s[judg- half_j][judg ]*p_s[judg- half_j][judg] ).sum()/2)
                    w_new = torch.min(t_1,w_new/w_old)
                    w_old = torch.max(t_0,1-w_new/w_old)
                    #print(judg-half_j,"to",judg,"  new:",ww_new," old:",ww_old,"  w_new:",nets_loss[judg],"  w_old:",nets_loss[judg-half_j])
                A[all, 0] = A[all, 0] * w_new / (w_new + w_old)
            #print(A[all,0])
  
        # 2. Sampling by acceptance rate
        B = A.reshape(-1)
        B = torch.where(torch.isnan(B), torch.full_like(B, 1), B)
        B = torch.where(torch.isinf(B), torch.full_like(B, 1), B)
        I = torch.multinomial(B, 1, replacement=True).data
        return I
    def fit(self, num_steps=1000,step_size=0.1):
        trajectory = []
        print("-------PMP HMC Optimizer------ ")
        tree_deep = math.log2(self.N + 1)
        # parameters, number of networks
        proposal_nets = [1] * (self.N + 1) # store each network
        p_s = torch.empty(self.N+1,self.N+1,self.d).to(device) # store the speed of the network p :p[i][j] indicates the speed of i when used for i to j
        nets_loss = [1] * (self.N + 1) # store the loss of each network

        for s in tqdm(range(num_steps)):
            proposal_nets[0] = copy.deepcopy(self.net).to(device) 
            # Initialize to generate N proposal networks (can be parallelized)
            for i in range(int(tree_deep)):
                j = int(math.pow(2, i))
                for k in range(int(j)):
                    p_s[k][k+j] = torch.randn(self.d).to(device) * 0.0005 # Randomly choose the original speed
                    proposal_nets[k+j] = copy.deepcopy(proposal_nets[k]).to(device)
                    p_s[k+j][k]= copy.deepcopy(p_s[k][k+j])

                    # Calculate potential energy  U(x)=-log(p(x))
                    yhat = proposal_nets[k](X)
                    nets_loss[k] = - self.loss(yhat, y)
                    # Calculate du_dx
                    nets_loss[k].backward()
                    du_dx = torch.tensor([]).to(device)
                    for par in proposal_nets[k].parameters():
                        par = par.grad.reshape(-1)
                        shape = par.shape[0]
                        du_dx = torch.cat([du_dx, par.reshape(shape, 1)])
                    du_dx = du_dx.reshape(self.d)
                    # leapfrog dynamic iteration
                    # 1. P take a half step
                    p_s[k+j][k] +=  step_size * du_dx / 2
                    # 2. Parameters go one step
                    sum = 0
                    for par in proposal_nets[k+j].parameters():
                        size = par.numel()
                        par.data += step_size * p_s[k+j][k][sum:sum + size].reshape(par.data.shape)
                        sum += size
                        # 3.Update the parameters required in the second half of the step
                    yhat = proposal_nets[k+j](X)
                    nets_loss[k+j] = - self.loss(yhat, y)
                    nets_loss[k+j].backward()
                    du_dx = torch.tensor([]).to(device)
                    for par in proposal_nets[j+k].parameters():
                        par = par.grad.reshape(-1)
                        shape = par.shape[0]
                        du_dx = torch.cat([du_dx, par.reshape(shape, 1)])
                    # 4. take half a step
                    du_dx = du_dx.reshape(self.d)
                    p_s[k+j][k] +=  step_size * du_dx / 2
            I = self.step(s,p_s,nets_loss,tree_deep)
            self.net = proposal_nets[I]
            self.loss_list.append(- nets_loss[I].cpu().item())
            if s%50==0:
                print("epoch",s,"loss = ",- nets_loss[I].cpu().item())
                correct = (self.net(X).argmax(1) == y).type(torch.float).sum().item()
                self.train_acc.append(correct / 60000)
                print("train_acc:",correct / 60000)
            
                correct = (self.net(x_test).argmax(1) == y_test).type(torch.float).sum().item()
                self.test_acc.append(correct / 10000)
                print("test_acc:",correct / 10000)

              
            # 记录每个参数的值
            parameters = torch.cat([param.view(-1) for param in self.net.parameters()])
            trajectory.append(parameters.cpu()[:10].detach().numpy().tolist())
        return np.array(self.loss_list),np.array(self.train_acc), np.array(self.test_acc),trajectory
model_PMPhmc = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=28*28, out_features=1024),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1024, out_features=10),    
    )
model_PMPhmc.load_state_dict(torch.load('./bnn_mnist.pth'))
model_PMPhmc = model_PMPhmc.to(device)
alpha = 0.001
num_steps = 30000

trainer = bnnPMPHmc(net = model_PMPhmc, alpha=alpha,N=N)
loss,train_acc,test_acc,samples = trainer.fit(num_steps=num_steps)

# save data
np.save('mnist_PMPhmc_loss_N_'+str(N)+'.npy', loss)
np.save('mnist_PMPhmc_train_acc_N_'+str(N)+'.npy', train_acc)
np.save('mnist_PMPhmc_test_acc_N_'+str(N)+'.npy', test_acc)
np.save('mnist_PMPhmc_samples_N_'+str(N)+'.npy', samples)
