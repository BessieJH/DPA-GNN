import math
import numpy as np
import torch
import torch.nn as nn
import time
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import generate_random_vectors, three_party,three_party_nonoise,gaussian_mechanism,gaussian_noise
# from Compiler import mpc_math
# from Compiler.types import sint
import gc
from datetime import datetime
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 为每个方设置一个唯一的标识符
# 例如，A=0, B=1, C=2
# party_id_A = 03
# party_id_B = 1
# party_id_C = 2
# bob_device = sy.Device(name="Bob's iPhone")

class GraphConvolution(Module):

    # 初始化层：输入feature，输出feature，权重，偏移
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # FloatTensor建立tensor

        # self.weight = nn.Parameter(self.weight.to('cuda:0'))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            # Parameters与register_parameter都会向parameters写入参数，但是后者可以支持字符串命名
        self.reset_parameters()


    # 初始化权重
    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        self.weight.data.uniform_(-stdv, stdv)  # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    '''
    forward代码: 
    '''
    def forward(self, input_features, edges):
        # 创建一个空列表用于存储字典
        list_of_dicts = []
        for row in edges:  # 遍历列表中的二维数组
            # 创建一个空字典
            new_dict = {}
            if row[1].item() == input_features.shape[0]:  # 检查二维数组的第二列（值）是否等于2708
                new_dict[row[0].item()] = torch.zeros(input_features.shape[1])  # 将值修改为0
            else:
                # 不做修改，直接存储键值对
                new_dict[row[0].item()] = input_features[row[1]]
            # 将字典添加到字典列表中
            list_of_dicts.append(new_dict)

         # 初始化存储秘密分享向量的列表
        shares_1 = []
        shares_2 = []
        shares_3 = []
        # 处理每个字典
        for d in list_of_dicts:
            for key, value in d.items():
                shared_vectors = generate_random_vectors(value)
                assert len(shared_vectors) == 3
                shares_1.append({key: shared_vectors[0]})
                shares_2.append({key: shared_vectors[1]})
                shares_3.append({key: shared_vectors[2]})


        shares_1 = three_party_nonoise(shares_1)   #不加噪
        shares_2 = three_party_nonoise(shares_2)
        shares_3 = three_party_nonoise(shares_3)


        epsilon = 8
        sensitivity = 1.0
        alpha = 1.1
        features_dim = 1433

        noise = gaussian_noise(shares_1, sensitivity, epsilon)

        # with open('output.txt', 'a') as f:
        #     print('shares_DP.shape:', shares_1.shape, shares_2.shape,shares_3.shape, file=f)
        #     print('noise.shape:', noise[0].shape, noise[1].shape, noise[2].shape, file=f)
        #     print('shares_1[0]+shares_2[0]+shares_3[0]:', shares_1[0]+shares_2[0]+shares_3[0], file=f)
        #     print('(noise[0]+noise[1]+noise[2])[0]:', (noise[0] + noise[1] + noise[2])[0], file=f)


        if len(shares_1[0]) == features_dim:  # 值的维度大于隐藏层维度
            shares_1_DP = shares_1 + noise[0]
            shares_2_DP = shares_2 + noise[1]
            shares_3_DP = shares_3 + noise[2]
        else:
            shares_1_DP = shares_1
            shares_2_DP = shares_2
            shares_3_DP = shares_3

        sum = shares_1_DP + shares_2_DP + shares_3_DP
        # sum = shares_1 + shares_2 + shares_3

        # with open('output.txt', 'a') as f:
        #     print('sum.shape:', sum.shape, file=f)
        #     print('sum[0]:', sum[0], file=f)
        #     print('sum[0]:', sum[0], file=f)


        sum = torch.nn.functional.normalize(sum, p=2, dim=1)   #归一化

        DP_values_list = sum.to(self.weight.dtype)   #使用to()方法，sum被转换为与self.weight.dtype相同的类型
        DP_values_list = DP_values_list.to(device)
        self.weight = self.weight.to(device)

        output = torch.mm(DP_values_list, self.weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

