from __future__ import division
from __future__ import print_function

import time
import argparse  # argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy
from model import GCN
import matplotlib.pyplot as plt

# pygcn代码解析  https://zhuanlan.zhihu.com/p/78191258?utm_id=0

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
args = parser.parse_args(args=[])
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


np.random.seed(args.seed)
torch.manual_seed(args.seed)   # 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
features, edges, labels, idx_train, idx_val, idx_test = load_data()


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


# 数据写入cuda，便于后续加速
if args.cuda:
    model.cuda()   # . cuda()会分配到显存里（如果gpu可用）
    features = features.cuda()
    edges = edges.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


current_time = datetime.now()  # 获取当前时间
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")  # 格式化当前时间


with open('output.txt', 'w') as f:
    print("当前时间:", formatted_time, file=f)


acc_train_plt = []
acc_val_plt = []
acc_test_plt = []
loss_train_plt = []
loss_val_plt = []
loss_test_plt = []


def train(epoch):
    # 定义训练的设备
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")  # 选择可用的GPU设备
    #     print("代码正在GPU上运行")
    # else:
    #     device = torch.device("cpu")  # 没有GPU可用，选择CPU设备
    #     print("代码正在CPU上运行")

    # features, edges, labels, idx_train, idx_val, idx_test = load_data()

    t = time.time()  # 返回当前时间
    model.train()
    optimizer.zero_grad()
    # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    # pytorch中每一轮batch需要设置optimizer.zero_gra
    output = model(features, edges)
    output = output.to(device)
    # labels = labels.to(device)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])  #计算准确率
    loss_train.backward()  # 反向求导  Back Propagation
    optimizer.step()  # 更新所有的参数  Gradient Descent

    # 验证，在评估模式下，模型的dropout层会被停用，以避免在验证过程中的随机性对结果的影响'，但是跑的时候耗时，暂时关闭了
    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     '将模型切换到评估模式（model.eval()）。在评估模式下，模型的dropout层会被停用，以避免在验证过程中的随机性对结果的影响'
    #     output = model(features, edges)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])    # 验证集的损失函数
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # 测试也随着epoch
    # model.eval()
    # output = model(features, edges)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # 在训练过程中也把测试精度打印出来
    print('Epoch: {:04d}'.format(epoch+1),
          "Train set results:",
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          "Validation set results:",
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))

    with open('output.txt', 'a') as f:
        print('Epoch: {:04d}'.format(epoch + 1),
              "Train set results:",
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              "Validation set results:",
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              "Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),
              'time: {:.4f}s'.format(time.time() - t),file=f)

    # 可视化  参考：https://www.icodebang.com/article/251200.html
    acc_train_plt.append(acc_train.item())
    acc_val_plt.append(acc_val.item())
    acc_test_plt.append(acc_test.item())
    loss_train_plt.append(loss_train.item())
    loss_val_plt.append(loss_val.item())
    loss_test_plt.append(loss_test.item())


# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
def end_result():
    model.eval()
    output = model(features, edges)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    with open('output.txt', 'a') as f:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),file=f)
    return acc_test


# Train model  逐个epoch进行train，最后test
t_total = time.time()
print("当前时间:", formatted_time)
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
with open('output.txt', 'a') as f:
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total),file=f)


# 测试最后一轮结果
end_result()

# 可视化  参考：https://www.icodebang.com/article/251200.html
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc_train_plt, label='Training Accuracy')
plt.plot(acc_val_plt, label='Validation Accuracy')
plt.plot(acc_test_plt, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation and Test Accuracy')
plt.subplot(1, 2, 2)
plt.plot(loss_train_plt, label='Training Loss')
plt.plot(loss_val_plt, label='Validation Loss')
plt.plot(loss_test_plt, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('plot.png')
plt.show()





