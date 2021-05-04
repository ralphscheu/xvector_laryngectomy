#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from model_utils import *


class xvecTDNN(nn.Module):
    """ Baseline x-vector model using statistics pooling (mean+std) """

    def __init__(self, numSpkrs, p_dropout):
        super(xvecTDNN, self).__init__()
        self.tdnn1 = tdnn_layer(in_channels=30, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=1)
        self.tdnn2 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=2)
        self.tdnn3 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=7, dilation=3)
        self.tdnn4 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=1, dilation=1)
        self.tdnn5 = tdnn_layer(in_channels=512, out_channels=1500, p_dropout=p_dropout, kernel_size=1, dilation=1)

        self.fc1 = fc_embedding_layer(in_channels=3000, out_channels=512, p_dropout=p_dropout)
        self.fc2 = fc_embedding_layer(in_channels=512, out_channels=512, p_dropout=p_dropout)
        self.fc3 = nn.Linear(512, numSpkrs)

    def forward(self, x, eps):
        """Note: x must be (batch_size, feat_dim, chunk_len)"""
        
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)  # statistics pooling
        x = self.fc1(stats)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class xvecTDNN_MHAttn(nn.Module):
    """ X-Vector model using Multihead Attention Pooling """

    def __init__(self, numSpkrs, p_dropout):
        super().__init__()
        self.tdnn1 = tdnn_layer(in_channels=30, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=1)
        self.tdnn2 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=2)
        self.tdnn3 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=7, dilation=3)
        self.tdnn4 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=1, dilation=1)
        self.tdnn5 = tdnn_layer(in_channels=512, out_channels=1500, p_dropout=p_dropout, kernel_size=1, dilation=1)

        self.attn_input_size = 1500
        self.mh_attn = MultiHeadAttention(
            key_size=self.attn_input_size, query_size=self.attn_input_size, value_size=self.attn_input_size,
            num_hiddens=1500, num_heads=1
        )

        self.fc1 = fc_embedding_layer(in_channels=1500, out_channels=512, p_dropout=p_dropout)
        self.fc2 = fc_embedding_layer(in_channels=512, out_channels=512, p_dropout=p_dropout)
        self.fc3 = nn.Linear(512,numSpkrs)

    def forward(self, x, eps):
        """Note: x must be (batch_size, feat_dim, chunk_len)"""

        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # add noise
        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps

        # apply Multihead Attention Pooling
        x = torch.moveaxis(x, 1, 2)
        x = self.mh_attn(x, x, x)
        x = torch.sum(x, dim=1)  # compute sum of weighted values per frame
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
