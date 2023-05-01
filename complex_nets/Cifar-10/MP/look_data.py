import numpy as np
loss = np.load("cifar_mphmc_loss.npy")
train_acc = np.load("cifar_mphmc_train_acc.npy")*100
test_acc = np.load("cifar_mphmc_test_acc.npy")*100
import matplotlib.pyplot as plt

sample_num = loss.shape[0]
line_loss = np.linspace(0,sample_num,sample_num)*2


sample_num = train_acc.shape[0]
line_acc = np.linspace(0,sample_num,sample_num)*2

fig, ax1 = plt.subplots()

# 纵坐标一
ax1.plot(line_loss, loss,"-r")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("loss")
ax1.set_ylim(0,2.4)
ax1.set_title("MP-HMC")

# 纵坐标二
ax2 = ax1.twinx()
ax2.plot(line_acc, train_acc,"-b")
ax2.set_ylabel("accuracy : %")
ax2.plot(line_acc, test_acc,"orange")
ax2.set_xlabel("Iterations")
fig.legend(["loss", "train", "test"],bbox_to_anchor=(0.8,0.5), loc="upper center")
plt.savefig("MP_HMC.png")
plt.show()
