#!/usr/bin/env python
# encoding: utf-8
# File Name: gat.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/18 21:40
# TODO:

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.gat import MultiHeadGraphAttention

class Gat(nn.Module):
    def __init__(self, embedding, units, heads, dropout, vertex_feature):
        super(Gat, self).__init__()
        self.layer_cnt = len(units) - 1
        self.dropout = dropout
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)
        self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
        self.vertex_feature.weight = nn.Parameter(vertex_feature)
        units[0] += embedding.size(1) + vertex_feature.size(1)

        self.layers = nn.ModuleList()
        for i in range(self.layer_cnt):
            in_features = units[i] * heads[i - 1] if i else units[i]
            self.layers.append(MultiHeadGraphAttention(heads[i], in_features=in_features, out_features=units[i + 1]))

    def forward(self, x, vertices, adj):
        emb = self.embedding(vertices)
        x = torch.cat((x, emb), dim=2)
        vfeature = self.vertex_feature(vertices)
        x = torch.cat((x, vfeature), dim=2)
        bs, n = adj.size()[:2]
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i + 1 == self.layer_cnt:
                x = x.mean(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
