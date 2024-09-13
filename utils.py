import numpy as np
import scipy.sparse as sp
import torch
import random
import math
from scipy.stats import geom

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_cora(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    edges = edges.tolist()
    edges = swap_and_append(edges)
    print("edges_swap:",len(edges))
    for i in range(0, features.shape[0]):
        data = [i, i]
        edges.append(data)
    print("edges_self:",len(edges))


    output_list = []
    for pair in edges:
        output_list.append({pair[0]: pair[1]})
    dgree_max = max_dgree(output_list, 0.9)


    edges = reduce_dict_count(output_list, dgree_max)
    edges = [[list(d.keys())[0], list(d.values())[0]] for d in edges]
    print("edges_sample:", len(edges))


    edges = dummy(0.5, edges, features)
    print("edges_dummy:", len(edges))


    dgree = dgree_matrix(edges, features)
    dgree = sp.coo_matrix(dgree)
    dgree = sparse_mx_to_torch_sparse_tensor(dgree)

    edges = torch.tensor(edges)


    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0]+1, labels.shape[0]+1),
                        dtype=np.float32)
    adj = sparse_mx_to_torch_sparse_tensor(adj)



    new_row_data = np.zeros((1, features.shape[1]), dtype=np.float32)
    new_row = sp.csr_matrix(new_row_data)
    features = sp.vstack((features, new_row))
    features = torch.FloatTensor(np.array(features.todense()))
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    labels = torch.LongTensor(np.where(labels)[1])



    adj = adj.to(torch.float64)
    features = features.to(torch.float64)
    dgree = dgree.to(torch.float64)

    features = torch.spmm(adj, features)

    epsilon = 1   #cora
    sensitivity = math.sqrt(dgree_max)

    noise = gaussian_noise(features, sensitivity, epsilon)
    features = features + noise
    features = torch.spmm(dgree, features)

    features = torch.spmm(adj, features)
    features = torch.spmm(dgree, features)

    features = torch.spmm(adj, features)
    features = torch.spmm(dgree, features)


    num = features.shape[0] - 1
    train_rate = 0.5
    val_rate = 0.75
    numbers = list(range(num))
    random.shuffle(numbers)
    idx_train = numbers[:int(num * train_rate)]  # 50%
    idx_val = numbers[int(num * train_rate):int(num * val_rate)]  # 25%
    idx_test = numbers[int(num * val_rate):num]  # 25%

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, dgree, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_1(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def generate_random_vectors(tensor_list):
    result1 = []
    result2 = []
    result3 = []

    for tensor in tensor_list:
        shape = tensor.shape

        flattened_vector = tensor.view(-1)
        # flattened_vector = flattened_vector.to(device)
        #
        # vector_1 = torch.randn(flattened_vector.shape).to(device)
        # vector_2 = torch.randn(flattened_vector.shape).to(device)

        vector_3 = flattened_vector - vector_1 - vector_2

        vector_1 = vector_1.view(shape)
        vector_2 = vector_2.view(shape)
        vector_3 = vector_3.view(shape)

        result1.append(vector_1)
        result2.append(vector_2)
        result3.append(vector_3)

    result1 = torch.stack(result1, dim=0)
    result2 = torch.stack(result2, dim=0)
    result3 = torch.stack(result3, dim=0)

    # print(len(result1), result1.size())
    # print(len(result2), result2.size())
    # print(len(result3), result3.size())

    return result1, result2, result3

def gaussian_noise(tensor1, sensitivity, epsilon):

    # tensor1 = tensor1.to(device)
    delta_f = sensitivity * math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon
    noise = torch.tensor(np.random.normal(loc=0, scale= delta_f, size=tensor1.shape))
    # noise = noise.to(device)

    return noise

def RDP_gaussian_noise(tensor1, sensitivity, epsilon):

    # tensor1 = tensor1.to(device)
    alpha = 20
    sigma = np.sqrt((sensitivity**2 * alpha) / (2*epsilon))
    noise = torch.tensor(np.random.normal(loc=0, scale= sigma/3, size=tensor1.shape))  #生成均值为0、标准差为sigma的高斯噪声
    return noise

def dummy(p,A,input_features):
    random_number = geom.rvs(p, size=input_features.shape[0])

    for i, num in enumerate(random_number):
        sub_array = [i, input_features.shape[0]]
        A.extend([sub_array] * num)

    return A



def swap_and_append(arr):
    new_arr = []
    for array in arr:
        if len(array) == 2:
            new_array = [array[1], array[0]]
            new_arr.append(new_array)
        new_arr.append(array)
    return new_arr

def swap_dummy(arr):
    new_arr = []
    for array in arr:
        if len(array) == 2:
            new_array = [array[1], array[0]]
            new_arr.append(new_array)
    return new_arr



def max_dgree(list_of_dicts, rate):
    count_dict = {}
    for d in list_of_dicts:
        for key in d.keys():
            count_dict[key] = count_dict.get(key, 0) + 1

    count_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[0])}
    # print(count_dict)
    dgree = list(count_dict.values())
    # print(dgree)
    sorted_list = sorted(dgree)
    # print(sorted_list)
    max_value = max(dgree)
    print("max_value:", max_value)
    index = int(rate * len(sorted_list))
    # print(index)
    result = sorted_list[index]
    print("max_dgree:",result)
    return result

def dgree_matrix(edges,input_fea):
    output_list = []
    for pair in edges:
        output_list.append({pair[0]: pair[1]})

    count_dict = {}
    for d in output_list:
        for key in d.keys():
            count_dict[key] = count_dict.get(key, 0) + 1
    # print(count_dict)
    count_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[0])}

    diagonal_matrix = torch.zeros(input_fea.shape[0]+1, input_fea.shape[0]+1)

    for key, value in count_dict.items():
        diagonal_matrix[key, key] = 1.0 / value

    return diagonal_matrix

def reduce_dict_count(input_list, dgree):
    positions_dict = {}
    new_value = ''
    # print(len(input_list))
    for index, dictionary in enumerate(input_list):
        key = list(dictionary.keys())[0]
        if key not in positions_dict:
            positions_dict[key] = []
        positions_dict[key].append(index)

    for key, positions in positions_dict.items():
        # print("Key:", key)
        # print("Positions:", positions)

        if len(positions) > dgree:
            for k in range(len(positions) - dgree):
                random_index = random.choice(positions)
                # input_list.pop(random_index)
                input_list[random_index] = new_value
                positions.remove(random_index)
                # print(random_index)
                # print(positions)
    # print(input_list)
    new_list = [x for x in input_list if x != '' and x is not None and len(str(x)) > 0]
    # print(new_list)
    return new_list