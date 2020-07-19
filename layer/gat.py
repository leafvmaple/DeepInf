from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, heads, in_features, out_features):
        super(MultiHeadGraphAttention, self).__init__()
        self.heads = heads
        self.w = Parameter(torch.Tensor(heads, in_features, out_features))
        self.a_src = Parameter(torch.Tensor(heads, out_features, 1))
        self.a_dst = Parameter(torch.Tensor(heads, out_features, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.bias = Parameter(torch.Tensor(out_features))
        init.constant_(self.bias, 0)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(F.tanh(h_prime), self.a_src)
        attn_dst = torch.matmul(F.tanh(h_prime), self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1)
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)
        output = torch.matmul(attn, h_prime)
        return output + self.bias