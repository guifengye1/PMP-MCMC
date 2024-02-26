# A Prefetching Multi-Proposal Parallel MCMC Algorithm

This repository is the official implementation of [A Prefetching Multi-Proposal Parallel MCMC Algorithm]() . 



## Requirements

To install requirements:
python=3.6
```setup
pip install -r requirements.txt
```
note: As for the code that using tensorflow, We should install a  tensorflow==2.10 and run it:
```setup
pip install --upgrade tensorflow-probability
```


Note: `.cu` files to run on NVIDIA Tesla A100.



## EX1：Effective Sample Size (Ess）and Error 


run `simple_sampling/error.py` to show error in different algorithms. The  result is shown in [error.pdf](https://github.com/guifengye1/PMP-MCMC/blob/main/simple_sampling/error/error.pdf).

![](https://img-blog.csdnimg.cn/direct/ad973b8ec90f4e97a84f5b1ca7a827f5.png)

Use SP-MCMC, MH-MCMC, and PMP-MCMC to sample from the banana distribution. The sampling effect is as follows:[banana.pdf](https://github.com/guifengye1/PMP-MCMC/tree/main/simple_sampling/error/banana/bananav3.pdf)
![](https://img-blog.csdnimg.cn/direct/0ff4559140704ffa92751ab90efd5065.png)
## EX2：Robustness and Convergence in simple model
### Robustness
run `simple_net/lb.py` to show the robustness in simple model. The  result is shown in [lb.pdf](https://github.com/guifengye1/PMP-MCMC/blob/main/simple_net/lb.pdf).
![](https://img-blog.csdnimg.cn/aafb9de61dc04975a541175ad13c9764.png)
### Time_analysis
We analyzed the acceleration effect of different complex problems, taking the data size of 500 and 100,000 as examples.
|Algorithm| Data size |Number of recommendations & number of cores |GPU run time per iteration : μs|Additional overhead time (host data processing + host and device dada transmission):μs|
| ------------------ |---------------- | -------------- |------------------ |------------------ |
| MP   |500| 4  | 157.505| 115.84
|MP| 500 | 1024 |452.258 | 1066.212
|MP| 100000 | 4 | 33465.447 | 346.528
|MP |100000 | 1024 | 33473.53|1099.258
| PMP   |500| 4  | 156.927| 65.952
|PMP| 500 | 1024 |177.952| 1289.7
|PMP| 100000 | 4 | 40259.869 | 538.464
|PMP |100000 | 1024 | 42096.793|2041.279
 
### Comparison of convergence speed of PMP, MP and SP
![](https://img-blog.csdnimg.cn/direct/17f9dd2fcad34193ac4f106e6462f0bf.png)
### Detailed analysis of PMP and MP convergence speed 
Comparison of convergence speeds of PMP and MP under different degrees of parallelism
![image](https://github.com/guifengye1/PMP-MCMC/assets/66373832/e7aad937-2c28-4da2-baa1-6047d1c6a106)

### ESS/s and MSJD/s
![image](https://github.com/guifengye1/PMP-MCMC/assets/66373832/e97b1c78-b57d-49be-939d-51bf1bf92456)


## EX3：correlation
We performed an investigation on the correlation between PMP-MCMC and the model's dimension d, chain length C, and prefetch depth D. Our study consisted of 125 tuples (d, D, C) with d ranging from 10 to 50, C ranging from 50 to 250, and D ranging from 1 to 5.
```python
python complex_nets/correlation/com_dim.py
```
The  result is shown in dimension_Chins_Parl.csv.
| *d=10*| D=1 |D=2 |D=3|D=4 | D=5|
| ------------------ |---------------- | -------------- |------------------ |------------------ | ------------------|
| C=50   |   1.005±1.092  |  0.540±1.141  | 0.671±1.157| 0.346±1.059|0.229±1.403 |
|C=100 | 0.430 ±1.273 | 0.406 ±1.106 |0.309 ±1.059 | 0.233 ±0.977 | 0.190 ±1.193  
|C=150 | 0.143 ±1.268 | 0.366 ±1.117 | 0.236 ±1.081 | 0.242 ±0.9256 | 0.157 ±1.141  
|C=200 | 0.089 ±1.224 | 0.235 ±1.125 | 0.114 ±1.084 | 0.161 ±0.967 | 0.114 ±1.098   
|C=250 | 0.066 ±1.189 | 0.259 ±1.068 | 0.110 ±1.073| 0.077 ±0.978 | 0.108 ±1.064   		
|*d=20* |  |  |  |  |  |
|C=50 | 1.920 ±1.141 | 1.253 ±1.323 | 1.131 ±1.466 | 0.679 ±1.244 | 0.278 ±1.282   
|C=100 | 1.166 ±1.363 | 0.710 ±1.320 | 0.660 ±1.335 | 0.269 ±1.142 | 0.087 ±1.161   
|C=150 | 0.758 ±1.377 | 0.596 ±1.233 | 0.448 ±1.285 | 0.207 ±1.079 | 0.094 ±1.121   
|C=200 | 0.534 ±1.358 | 0.541 ±1.170 | 0.340 ±1.216 | 0.195 ±1.068 | 0.068 ±1.111   
|C=250 | 0.432 ±1.314| 0.424 ±1.139 | 0.289 ±1.170 | 0.166 ±1.050 | 0.087 ±1.085   
|*d=30* |  |  |  |  |    
|C=50 | 1.542 ±1.556 | 1.597 ±1.285 | 1.3241 ±1.493 | 0.988 ±1.418 | 0.745 ±1.510  
|C=100 | 1.151 ±1.722 | 1.014 ±1.471 | 0.613 ±1.573 | 0.396 ±1.463 | 0.377 ±1.391  
|C=150 | 0.904 ±1.763 | 0.712 ±1.463 | 0.434 ±1.478 | 0.241 ±1.368 | 0.220 ±1.284  
|C=200 | 0.679 ±1.747 | 0.506 ±1.441 | 0.326 ±1.370 | 0.159 ±1.283 | 0.151 ±1.231  
|C=250 | 0.527 ±1.690 | 0.451 ±1.362 | 0.254 ±1.308 | 0.146 ±1.224 | 0.127 ±1.203  
|*d=40* |  |  |  |  | 
|C=50 | 2.349 ±0.693 | 1.916 ±1.233 | 1.439 ±1.513 | 1.436 ±1.406 | 1.171 ±1.449  
|C=100 | 2.054 ±1.131 | 1.402 ±1.489 | 1.074 ±1.555 | 0.925 ±1.527 | 0.707 ±1.458  
|C=150 | 1.687 ±1.415 | 1.152 ±1.512 | 0.891 ±1.523 | 0.652 ±1.489 | 0.508 ±1.404  
|C=200 | 1.391 ±1.552 | 0.916 ±1.554 | 0.749 ±1.504 | 0.498 ±1.435 | 0.346 ±1.366  
|C=250 | 1.185 ±1.599 | 0.766 ±1.544 | 0.650 ±1.479 | 0.417 ±1.385 | 0.288 ±1.332  
|*d=50* | | | |  |   
|C=50 | 2.287 ±0.912 | 2.265 ±0.853 | 1.956 ±1.350 | 1.991 ±1.315 | 1.857 ±1.410  
|C=100 | 2.155 ±1.113 | 1.967 ±1.241 | 1.688 ±1.590 | 1.685 ±1.604 | 1.349 ±1.668  
|C=150 | 2.046 ±1.224 | 1.700 ±1.503 | 1.449 ±1.717 | 1.439 ±1.733 | 1.068 ±1.717  
|C=200 | 1.970 ±1.292 | 1.491 ±1.643 | 1.227 ±1.804 | 1.251 ±1.751 | 0.857 ±1.732  
|C=250 | 1.892 ±1.355 | 1.324 ±1.717 | 1.051 ±1.848 | 1.116 ±1.753 | 0.720 ±1.729  


## EX4：Robustness and Convergence in complex models

### Robustness
For FC model
```train
python ./complex_nets/Mnist/FC/MH_FC.py
python ./complex_nets/Mnist/FC/MP_FC.py
python ./complex_nets/Mnist/FC/PMP_FC.py
```
For CNN model
```train
python ./complex_nets/Mnist/FC/MH_CNN.py
python ./complex_nets/Mnist/FC/MP_CNN.py
python ./complex_nets/Mnist/FC/PMP_CNN.py
```
For LSTM model
```train
python ./complex_nets/Mnist/FC/MH_LSTM.py
python ./complex_nets/Mnist/FC/MP_LSTM.py
python ./complex_nets/Mnist/FC/PMP_LSTM.py
```
The  result is shown in /complex_nets/Mnist/[result.png](https://github.com/guifengye1/PMP-MCMC/blob/main/complex_nets/Mnist/result.png).
![](https://img-blog.csdnimg.cn/4c6d6665ea264fae8d818b7b49e34dc0.png)
### Convergence
```train
python ./complex_nets/Cifar-10/cifar_MPhmc.py
python ./complex_nets/Cifar-10/cifar_PMPhmc.py
python ./complex_nets/Cifar-10/cifar_SPhmc.py
```
The  result is shown in /complex_nets/Cifar-10/[result.png](https://github.com/guifengye1/PMP-MCMC/blob/main/complex_nets/Cifar-10/result.png).


![](https://img-blog.csdnimg.cn/135e72bfbd48423f8fc4f9d402f9376c.png)

## Pre-trained Models

You can download pretrained models here:

cifar
- [cifar.pkl
](https://github.com/guifengye1/PMP-MCMC/blob/main/complex_nets/Cifar-10/cifar.pkl)

Mnist
- [FC](https://github.com/guifengye1/PMP-MCMC/blob/main/complex_nets/Mnist/FC/FC_model.pkl) 
- [LSTM](https://github.com/guifengye1/PMP-MCMC/blob/main/complex_nets/Mnist/LSTM/LSTM_model.pkl)
- [CNN](https://github.com/guifengye1/PMP-MCMC/tree/main/complex_nets/Mnist/CNN)



