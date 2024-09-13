from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import statistics
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import  accuracy,load_cora
from models import GCN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
import random

torch.set_printoptions(threshold=np.inf)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,   #10−4, 10−3, 10−2
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, #10−4, 10−3, 10−2
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# random.seed(args.seed)
# np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def run():
    # Load data
    adj, features, labels, dgr, idx_train, idx_val, idx_test = load_cora()

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()



    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].long())
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].long())
        acc_val = accuracy(output[idx_val], labels[idx_val])

        # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].long())
        acc_test = accuracy(output[idx_test], labels[idx_test])

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
        return loss_val

    def end_result():
        model.eval()
        output = model(features)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].long())
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output[idx_test], labels[idx_test]


    t_total = time.time()
    eval_T = 5  # evaluate period
    P = 9  # patience
    i = 0  # record the frequency of bad performance of validation
    temp_val_loss = 99999  # initialize val loss

    for epoch in range(args.epochs):
        result = train(epoch)
        # early stopping
        if (epoch % eval_T) == 0:
            if temp_val_loss > result:
                temp_val_loss = result
                # torch.save(model.state_dict(), "GCN_NET3.pth")  # save the current best
                i = 0  # reset i
            else:
                i = i + 1
        if i > P:
            print("Early Stopping! Epoch : ", epoch, )
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test_acc, result, test_label = end_result()

    return test_acc

    warnings.filterwarnings("ignore")

    def visualize_results(embeddings, labels):
        random.seed(0)
        embeddings = embeddings.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()
        # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)
        # tsne = TSNE(n_components=2, verbose=1, perplexity=20, learning_rate=200, n_iter=1000, random_state=0)
        tsne = TSNE(n_components=2, random_state=0)
        # tsne = TSNE(n_components=2, random_state=0)
        x = tsne.fit_transform(embeddings)
        yi = ['0', '1', '2', '3', '4', '5', '6']
        xi = []
        for i in range(7):
            xi.append(x[np.where(y == i)])

        colors = [ 'blue','red', 'green', 'orange', 'purple', 'pink', 'brown']
        plt.figure(figsize=(8, 6))
        # for i in range(7):
        #     plt.scatter(xi[i][:, 0], xi[i][:, 1], s=30, marker='*', alpha=1, label=str(i), c=colors[i])
        for i, data in enumerate(yi):
            plt.scatter(xi[i][:, 0], xi[i][:, 1], s=30, marker='o', alpha=1, label=str(i), c=colors[i % len(colors)])

        plt.legend()
        plt.xlabel('90%')
        plt.savefig('plot1.png')
        # plt.show()

    visualize_results(result,test_label)


run()




