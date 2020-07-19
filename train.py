from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.gcn import Gcn
from models.gat import Gat
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from utilst import load_data, get_data_info

import os
import shutil

embedding, vertex_features, train_data, valid_data, test_data = load_data(embedding_dim=64, train_ratio= 75, valid_ratio=12.5)
classes_cnt, class_weight, feature_dim = get_data_info(train_data)

def evaluate(model, epoch, data):
    model.eval()
    total = 0
    loss = 0
    y_true, y_pred, y_score = [], [], []
    for graph, features, labels, vertices in enumerate(data):
        cnt = graph.size(0)

        output = model(features, vertices, graph)
        output = output[:, -1, :]
        loss_batch = F.nll_loss(output[:, -1, :], labels, class_weight)
        loss += cnt * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += cnt

    model.train()

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)

    print("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f", loss / total, auc, prec, rec, f1)


def train(model_type, epochs):
    if model_type == "gcn":
        model = Gcn(embedding=embedding, units=[feature_dim, 64, 64, classes_cnt], dropout=0.2, vertex_feature=vertex_features)
    elif model_type == "gat":
        model = Gat(embedding=embedding, units=[feature_dim, 16, 16, classes_cnt], heads=[8, 8, 1], dropout=0.1, vertex_feature=vertex_features)

    params = [{'params': model.layers.parameters()}]
    optimizer = optim.Adagrad(params, lr=1e-3, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()

        loss = 0
        total = 0

        for graph, features, labels, vertices in enumerate(train_data):
            cnt = graph.size(0)

            optimizer.zero_grad()
            output = model(features, vertices, graph)
            loss_train = F.nll_loss(output[:, -1, :], labels, class_weight)
            loss += cnt * loss_train.item()
            total += cnt
            loss_train.backward()
            optimizer.step()

        print("epoch %d loss: %f", epoch, loss / total)

    evaluate(model, epochs, valid_data)
    evaluate(model, epochs, test_data)

train("gcn", 200)
train("gat", 300)


