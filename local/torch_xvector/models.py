#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math


def tdnn_layer(in_channels, out_channels, p_dropout, *args, **kwargs):
    return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, *args, **kwargs),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels, momentum=0.1, affine=False),
            nn.Dropout(p=p_dropout)
        )


def fc_embedding_layer(in_channels, out_channels, p_dropout, *args, **kwargs):
    return nn.Sequential(
            nn.Linear(in_channels, out_channels, *args, **kwargs),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels, momentum=0.1, affine=False),
            nn.Dropout(p=p_dropout)
        )


class xvector(nn.Module):
    """ Baseline x-vector model using statistics pooling (mean+std) """

    def __init__(self, numSpkrs, p_dropout=0, rank=0):
        super().__init__()
        self.rank = rank

        self.tdnn1 = tdnn_layer(in_channels=30, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=1)
        self.tdnn2 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=2)
        self.tdnn3 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=7, dilation=3)
        self.tdnn4 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=1, dilation=1)
        self.tdnn5 = tdnn_layer(in_channels=512, out_channels=1500, p_dropout=p_dropout, kernel_size=1, dilation=1)

        self.fc1 = fc_embedding_layer(in_channels=3000, out_channels=512, p_dropout=p_dropout)
        self.fc2 = fc_embedding_layer(in_channels=512, out_channels=512, p_dropout=p_dropout)
        self.fc3 = nn.Linear(512, numSpkrs)

    def forward(self, x, eps=1e-5):
        """Note: x must be (batch_size, feat_dim, chunk_len)"""
        
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape).to(self.rank)
            torch.randn(shape, out=noise)
            x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)  # statistics pooling
        x = self.fc1(stats)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class xvector_legacy(nn.Module):
    """
    DO NOT USE FOR TRAINING
    Legacy model without tdnn and fc layers factored out, effectively the architecture is the same.
    Baseline x-vector model using statistics pooling (mean+std)
    -> need for loading xvector__20210409T145152 <-
    """

    def __init__(self, numSpkrs, p_dropout):
        super().__init__()
        self.tdnn1 = nn.Conv1d(in_channels=30, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512,512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512,numSpkrs)

    def forward(self, x, eps):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x
