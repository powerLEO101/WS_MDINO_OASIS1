from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

def represent_class(args):
    train_path_feat_x = '/home/leo101/Work/Harp/dino/trainfeatx.npy'
    train_path_feat_y = '/home/leo101/Work/Harp/dino/trainfeaty.npy'
    train_path_feat_z = '/home/leo101/Work/Harp/dino/trainfeatz.npy'
    train_path_labels = '/home/leo101/Work/Harp/dino/trainlabelsz.npy'

    #X = torch.tensor(np.load(train_path_feat_z))
    X = torch.cat((torch.tensor(np.load(train_path_feat_y)), torch.tensor(np.load(train_path_feat_z)), torch.tensor(np.load(train_path_feat_x))), dim=-1)
    y = torch.tensor(np.load(train_path_labels))
    #EX = torch.tensor(StandardScaler().fit_transform(X, y))

    X_embedded_PCA = PCA(2).fit_transform(X)
    #X_embedded = TSNE(2, n_iter=5000).fit_transform(X_embedded_PCA)
    plt.scatter(X_embedded_PCA[:, 0], X_embedded_PCA[:, 1], c=y)
    #plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
    plt.savefig(args.save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('WS-DINO-represent')
    parser.add_argument('--save_name', default='fig.png', type=str, help='''Name for saving figure''')
    args = parser.parse_args()
    represent_class(args)
    