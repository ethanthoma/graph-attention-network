import os
import networkx as nx
import pandas as pd
import numpy as np
from tinygrad.tensor import Tensor

import scipy.sparse as sp


def fetch_cora(tensors=False):
    data_dir = "./data/cora"

    # fetch data from path
    cites_file = f"{data_dir}/cora.cites"
    edges_unordered = np.genfromtxt(cites_file, dtype=np.int32)

    content_file = f"{data_dir}/cora.content"
    idx_features_labels = np.genfromtxt(content_file, dtype=str)

    node_ids = idx_features_labels[:, 0].astype(np.int32)
    features = sp.csr_matrix(
        idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(
        labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(250)
    idx_test = range(250, 1500)

    adj = np.array(adj.todense())
    features = np.array(features.todense())
    labels = np.where(labels)[1]

    X_train = (features[idx_train], adj[idx_train, idx_train])
    Y_train = labels[idx_train]
    X_test = (features[idx_test], adj[idx_test, idx_test])
    Y_test = labels[idx_test]

    if tensors:
        X_train = (Tensor(X_train[0]), Tensor(X_train[1]))
        Y_train = Tensor(Y_train)
        X_test = (Tensor(X_test[0]), Tensor(X_test[1]))
        Y_test = Tensor(Y_test)

    return X_train, Y_train, X_test, Y_test


def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return adj.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
