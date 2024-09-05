import numpy as np
import torch
epsilon = 1
sensitivity = 1.0
alpha = 2
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)
sigma = np.sqrt((sensitivity ** 2 * alpha) / (2 * epsilon))
noise = torch.tensor(np.random.normal(loc=0, scale=sigma, size=(1,10)))  # 生成均值为0、标准差为sigma的高斯噪声

print(noise[0])