import numpy as np
from tqdm import tqdm

# [1285000000,1625000000,3220000000,5121000000]
# file_names = [1024500000]
file_names = [850000000]
file_names = [2562000000]
file_names = [8,16,32,64,128,256,512]
for file_name in file_names:
    print("load :",file_name)
    beta0 = []
    file_path = "./"+str(file_name)+'1000000_sigma_true.txt'
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            beta0.append(float(line.strip()))
    beta0 = np.array(beta0)
    np.save(str(file_name)+'1000000_sigma_true.npy',beta0)