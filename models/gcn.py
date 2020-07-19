from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from layer.gcn import GraphConvolution

class Gcn(nn.Module):
    def __init__(self, units, dropout, embedding, vertex_feature):
        super(Gcn, self).__init__()
        self.layer_cnt = len(units) - 1
        self.dropout = dropout
        self.norm = nn.InstanceNorm1d(embedding.size(1), momentum=0.0, affine=True)
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding))
        self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
        self.vertex_feature.weight = nn.Parameter(vertex_feature)
        units[0] += embedding.shape[1] + vertex_feature.size(1)

        self.layers = nn.ModuleList()

        for i in range(self.layer_cnt):
            self.layers.append(GraphConvolution(units[i], units[i + 1]))

    def forward(self, x, vertices, lap):
        emb = self.embedding(vertices)
        emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        vfeature = self.vertex_feature(vertices)
        x = torch.cat((x, vfeature), dim=2)
        for i, layer in enumerate(self.layers):
            x = layer(x, lap)
            if i + 1 < self.layer_cnt:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)