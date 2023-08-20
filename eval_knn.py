import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn

import utils
from torchsampler import ImbalancedDatasetSampler

def balance_class(X, y):
    count = np.bincount(y)
    weight = [None] * 3
    weight[1] = int(count[0] / count[1])
    weight[2] = int(count[0] / count[2])
    X_1 = []; y_1 = []
    for idx, label in enumerate(y):
        if label == 0:
            continue
        X_tmp = [X[idx]] * weight[label]
        y_tmp = [label] * weight[label]
        X_1 = X_1 + X_tmp
        y_1 = y_1 + y_tmp

    X_1 = torch.stack(X_1)
    y_1 = torch.stack(y_1)
    X_1 = torch.cat([X, X_1])
    y_1 = torch.cat([y, y_1])
    return X_1, y_1


def eval_knn():
    train_path_feat_x = '/home/leo101/Work/Harp/dino/trainfeatx.npy'
    train_path_feat_y = '/home/leo101/Work/Harp/dino/trainfeaty.npy'
    train_path_feat_z = '/home/leo101/Work/Harp/dino/trainfeatz.npy'
    train_path_labels = '/home/leo101/Work/Harp/dino/trainlabelsz.npy'
    batch_size = 32

    #X = torch.tensor(np.load(train_path_feat_x))
    #X = torch.cat((torch.tensor(np.load(train_path_feat_y)), torch.tensor(np.load(train_path_feat_x))), dim=-1)
    X = torch.cat((torch.tensor(np.load(train_path_feat_y)), torch.tensor(np.load(train_path_feat_z)), torch.tensor(np.load(train_path_feat_x))), dim=-1)
    y = torch.tensor(np.load(train_path_labels))
    y[y==2] = 1
    X = torch.tensor(StandardScaler().fit_transform(X, y))


    correct = 0

    t = [0] * 2
    all = [0] * 2

    for idx in range(len(X)):
        train_X = torch.cat([X[:idx, :], X[idx + 1:, :]], dim=0)
        train_y = torch.cat([y[:idx], y[idx + 1:]], dim=0)
        #train_X, train_y = balance_class(train_X, train_y)
        model = KNeighborsClassifier(2).fit(train_X, train_y)
        pred = model.predict(X[idx].reshape(1, -1))
        if int(pred) == y[idx].item():
            correct += 1
            t[int(pred)] += 1
        all[y[idx].item()] += 1
        #correct = correct + (int(pred) == y[idx].item())

    print(correct)
    print(correct / len(X), t[0] / all[0], t[1] / all[1])

if __name__ == '__main__':
    eval_knn()