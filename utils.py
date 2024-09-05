import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random
from collections import defaultdict
from collections import Counter
from scipy.stats import geom
from number import numbers3

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode_onehot(labels):
    classes = set(labels)  # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in  # identity创建方矩阵
                    enumerate(classes)}     # 字典 key为label的值，value为矩阵的每一行
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  # get函数得到字典key对应的value
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    # 以稀疏矩阵（采用CSR格式压缩）将数据中的特征存储
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))  # 将特征转换为tensor


    labels = encode_onehot(idx_features_labels[:, -1]) # 这里的label为onthot格式，如第一类代表[1,0,0,0,0,0,0]
    labels = torch.LongTensor(np.where(labels)[1])

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 将每篇文献的编号idx提取出来
    idx_map = {j: i for i, j in enumerate(idx)}

    # 读取cite文件，以二维数组的形式存储
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),  # flatten：降维，返回一维数组
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 将NumPy数组转换为列表
    edges = edges.tolist()
    edges = swap_and_append(edges)  #添加镜像数据，无向图
    for i in range(0, features.shape[0]):  # 添加自身（ki:xi)的信息
        data = [i, i]  # 创建一个包含[i, i]的数据
        edges.append(data)  # 将数据添加到列表中

    "造假数据"
    # 设置几何分布的参数
    p = 0.3  # 成功的概率
    # 使用几何分布的随机数生成器生成满足几何分布的数字
    random_number = geom.rvs(p, size=features.shape[0])  # 生成10个满足几何分布的随机数

    for i, num in enumerate(random_number):
        sub_array = [i, features.shape[0]]  # 生成满足要求的二维数组,在后面定义features.shape[0]为0
        edges.extend([sub_array] * num)  # 将二维数组根据数字出现的次数添加到结果列表中

    edges = torch.tensor(edges)


    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    idx_train = range(1600)
    idx_val = range(1601, 2140)
    idx_test = range(2141, 2708)

    '由于数据质量不均匀，所以打乱排序'
    # numbers = list(range(2708))  # 生成0到2707的数字
    # random.shuffle(numbers)  # 打乱顺序
    # idx_train = numbers3[:1600]   #60%
    # idx_val = numbers3[1601:2140]  #20%
    # idx_test = numbers3[2141:2708]  #20%

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    return features, edges, labels, idx_train, idx_val, idx_test


"生成镜像数据，无向图"
def swap_and_append(arr):
    new_arr = []  # 创建一个新的空列表用于存储交换后的数组
    for array in arr:
        if len(array) == 2:  # 只处理包含两个元素的数组
            new_array = [array[1], array[0]]  # 交换第一位数和第二位数的位置
            new_arr.append(new_array)  # 将交换后的数组添加到新的列表中
        new_arr.append(array)  # 将原始数组添加到新的列表中
    return new_arr


def three_party(dicts):
    "聚合key相同的列"
    # AGG_result = add_dicts(dicts)   #加和聚合
    AGG_result = ave_dicts(dicts)    #平均聚合
    # 按照键从小到大的顺序对字典列表进行排序
    AGG_result = sorted(AGG_result, key=lambda x: list(x.keys())[0])

    "删除所有的键，保留值 "
    values_list = process_dicts_list(AGG_result)

    with open('output.txt', 'a') as f:
        print('len(values_list):', len(values_list), file=f)
        print('values_list[0]:', values_list[0], file=f)

    epsilon =10
    sensitivity = 1.0
    alpha = 1.1
    features_dim = 1433
    "噪声判断在噪声的函数中"

    if len(values_list[0]) == features_dim:   #值的维度大于隐藏层维度
        # DP_values_list = gaussian_mech_RDP_vec(values_list, sensitivity, alpha, epsilon) #renyi差分隐私
        # DP_values_list = laplace_mech_RDP_vec(values_list, sensitivity, alpha, epsilon)
        DP_values_list = gaussian_mechanism(values_list, sensitivity, epsilon)
        # DP_values_list = laplace_mechanism(values_list, sensitivity, epsilon)
    else:
        DP_values_list = values_list

    with open('output.txt', 'a') as f:
        print('len(DP_values_list):', len(DP_values_list), file=f)
        print('DP_values_list[0]:', DP_values_list[0], file=f)


    return DP_values_list

#不加噪声，对比实验
def three_party_nonoise(dicts):

    "聚合key相同的列"
    AGG_result = ave_dicts(dicts)  #均值聚合
    # AGG_result = add_dicts(list_of_dicts)  #加和聚合
    # 按照键从小到大的顺序对字典列表进行排序
    AGG_result = sorted(AGG_result, key=lambda x: list(x.keys())[0])

    "删除所有的键，保留值 "
    values_list = process_dicts_list(AGG_result)
    return values_list

"加和-聚合相同key的值"
def add_dicts(dicts):
    # 创建一个空的字典来存储相加后的值
    result_dict = {}
    # 迭代每个字典
    for dictionary in dicts:
        for key, value in dictionary.items():

            # 如果键已经存在于结果字典中，将值相加
            if key in result_dict:
                result_dict[key] += value
            # 如果键不存在于结果字典中，将键和值添加到结果字典中
            else:
                result_dict[key] = value

   # 将结果字典转换为列表
    result_list = [{key: value} for key, value in result_dict.items()]
    return result_list

"平均-聚合相同key的值"
def ave_dicts(dicts):
    counts = defaultdict(int)
    sums = defaultdict(int)

    # 遍历列表中的字典
    for d in dicts:
        # 遍历字典中的键值对
        for key, value in d.items():
            # 统计键出现的次数
            counts[key] += 1
            # 对相同键的值进行累加
            sums[key] += value

    combined_dicts = []

    # 计算平均值并构建新的字典
    for key, count in counts.items():
        avg_value = sums[key] / count
        combined_dicts.append({key: avg_value})

    return combined_dicts

def supplement_key(dictionary_list,input_features):
    # 创建一个空的字典列表
    new_dictionary_list = []
    # 创建一个集合用于存储范围内已存在的键
    keys_set = set()
    # 获取第一个字典的值作为参考来确定张量的形状
    reference_value = next(iter(dictionary_list[0].values()))
    reference_shape = reference_value.shape

    # 遍历原始的字典列表
    for dictionary in dictionary_list:
        # 检查键是否在范围内
        for key in dictionary.keys():
            if key in range(input_features.shape[0]):
                # 将字典添加到新的字典列表中，并将键添加到集合中
                new_dictionary_list.append(dictionary)
                keys_set.add(key)


    for key in range(input_features.shape[0]):
        if key not in keys_set:
            # 创建一个字典，并将键添加到字典中，值设置为全为 0 的张量
            value = torch.zeros(*reference_shape)  # 修改为适当的维度
            value = value.to(device)
            dictionary = {key: value}
            # 将字典添加到新的字典列表中
            new_dictionary_list.append(dictionary)

    # 按照键从小到大的顺序对字典列表进行排序
    new_dictionary_list = sorted(new_dictionary_list, key=lambda x: list(x.keys())[0])

    return new_dictionary_list


def process_dicts_list(dicts):
    # 获取所有字典的值，并将其存储在一个列表中
    values_list = [list(dictionary.values()) for dictionary in dicts]
    new_tensor_list = [tensor[0] for tensor in values_list]
    # 将列表转换为张量矩阵
    tensor_matrix = torch.stack(new_tensor_list)

    return tensor_matrix


def generate_random_vectors(tensor):
    """
    Generates two random vectors by splitting a given tensor.

    :param tensor: The original tensor.
    :return: A list of two random vectors (torch tensors).
    """
    # Get the shape of the original tensor
    shape = tensor.shape
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Flatten the tensor to a 1D vector
    flattened_vector = tensor.view(-1)
    flattened_vector = flattened_vector.to(device)

    # Generate the first random vector
    vector_1 = torch.randn(flattened_vector.shape).to(device)
    vector_2 = torch.randn(flattened_vector.shape).to(device)

    # Calculate the second vector such that the sum of both vectors equals the original tensor
    vector_3 = flattened_vector - vector_1 - vector_2

    # Reshape the vectors back to the original shapezhoushen
    vector_1 = vector_1.view(shape)
    vector_2 = vector_2.view(shape)
    vector_3 = vector_3.view(shape)

    return [vector_1, vector_2, vector_3]

# def gaussian_mech_RDP_vec(tensor, sensitivity, alpha, epsilon):
#
#     tensor = tensor.to(device)
#     # 计算隐私预算epsilon对应的高斯噪声方差
#     sigma = np.sqrt((sensitivity**2 * alpha) / (2*epsilon))
#
#     # 生成与张量形状相同的高斯噪声
#     noise = torch.tensor(np.random.normal(loc=0, scale=sigma, size=tensor.shape))  #生成均值为0、标准差为sigma的高斯噪声
#     noise = noise.to(device)
#     # 将噪声添加到张量上
#     noisy_tensor = tensor + noise
#     return noisy_tensor

def gaussian_mech_RDP_vec(tensor_matrix, sensitivity, alpha, epsilon):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor_matrix = tensor_matrix.to(device)

    if tensor_matrix.shape[1] > 300:
        for i in range(tensor_matrix.shape[0]):
            selected_indices = np.random.choice(tensor_matrix.shape[1], size=300, replace=False)
            # sensitivity = np.max(np.abs(tensor_matrix[i, selected_indices]))
            sigma = np.sqrt((sensitivity ** 2 * alpha) / (2 * epsilon))
            noise = torch.tensor(np.random.normal(loc=0, scale=sigma, size=300))  #生成均值为0、标准差为sigma的高斯噪声

            # sensitivity = 1
            # 'sensitivity计算了被选择的数据的最大绝对值。它通过将tensor_matrix[i, selected_indices]的绝对值取最大值来得到。'
            # scale = sensitivity * (2 * np.log(1.25 / delta)) ** 0.5 / (epsilon * (alpha - 1.0) ** 0.5)
            # noise = torch.tensor(np.random.normal(loc=0.0, scale=scale, size=300))
            noise = noise.to(device)
            tensor_matrix[i, selected_indices] += noise
    else:
        tensor_matrix = tensor_matrix
    return tensor_matrix


def laplace_mech_RDP_vec(tensor, sensitivity, alpha, epsilon):

    tensor = tensor.to(device)
    # 计算隐私预算epsilon对应的高斯噪声方差
    sigma = np.sqrt((sensitivity**2 * alpha) / (2*epsilon))

    # 生成与张量形状相同的高斯噪声
    noise = torch.tensor(np.random.laplace(loc=0, scale=sigma, size=tensor.shape))  #生成均值为0、标准差为sigma的高斯噪声
    noise = noise.to(device)
    # 将噪声添加到张量上
    noisy_tensor = tensor + noise
    return noisy_tensor


def gaussian_mechanism(tensor1, sensitivity, epsilon):

    tensor1 = tensor1.to(device)
    delta_f = sensitivity / epsilon
    noise = torch.tensor(np.random.normal(0, delta_f, tensor1.shape))

    noise = noise.to(device)
    # 将噪声添加到张量上
    noisy_tensor = tensor1 + noise
    return noisy_tensor

def gaussian_noise(tensor1, sensitivity, epsilon):

    tensor1 = tensor1.to(device)
    delta_f = sensitivity / epsilon
    noise = torch.tensor(np.random.normal(0, delta_f, tensor1.shape))

    noise = noise.to(device)

    # 将噪声添加到张量上
    noise = generate_random_vectors(noise)
    return noise

def laplace_mechanism(tensor, sensitivity, epsilon):

    tensor = tensor.to(device)
    delta_f = sensitivity / epsilon
    noise = torch.tensor(np.random.laplace(0, delta_f, tensor.shape))

    noise = noise.to(device)
    # 将噪声添加到张量上
    noisy_tensor = tensor + noise
    return noisy_tensor

def normalize(mx):
    """Row-normalize sparse matrix"""

    rowsum = np.array(mx.sum(1))  # 对每一行求和,得到一个（2708,1）的矩阵
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数, 得到（2708，）的元组
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def accuracy(output, labels):
    # 将预测结果转换为和labels一致的类型
    preds = output.max(1)[1].type_as(labels) # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):    # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 这里我没咋搞懂怎么stack的 求指教
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


