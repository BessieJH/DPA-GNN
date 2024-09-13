import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#
#         self.fc1 = nn.Linear(nfeat, nhid)
#         self.fc2 = nn.Linear(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x.float()))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.fc2(x.float())
#         return F.log_softmax(x, dim=1)


