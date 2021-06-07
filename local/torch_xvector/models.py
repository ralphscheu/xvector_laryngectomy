#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math


#########################
#   MODEL COMPONENTS    #
#########################

def tdnn_layer(in_channels, out_channels, p_dropout, *args, **kwargs):
    return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, *args, **kwargs),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels, momentum=0.1, affine=False),
            nn.Dropout(p=p_dropout)
        )


def scaled_dot_product_attention(query, key, value, mask=None, p_dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    (https://nlp.seas.harvard.edu/2018/04/03/attention.html)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if p_dropout is not None:
        drop = nn.Dropout(p=p_dropout)
        p_attn = drop(p_attn)
    return torch.matmul(p_attn, value), p_attn


def transpose_qkv(X, num_heads):
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """
    Multihead Self Attention Layer
    adapted from https://d2l.ai/chapter_attention-mechanisms/multihead-attention.html#multi-head-attention
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, p_dropout=0.1, bias=False, **kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.W_q = nn.Linear(query_size,    num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size,      num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size,    num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens,   num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        """
        Shape of `queries`, `keys`, or `values`:
        (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        
        After transposing, shape of output `queries`, `keys`, or `values`:
        (`batch_size` * `num_heads`, no. of queries or key-value pairs, `num_hiddens` / `num_heads`)
        """

        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output, attn = scaled_dot_product_attention(queries, keys, values, p_dropout=self.p_dropout)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def fc_embedding_layer(in_channels, out_channels, p_dropout, *args, **kwargs):
    return nn.Sequential(
            nn.Linear(in_channels, out_channels, *args, **kwargs),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels, momentum=0.1, affine=False),
            nn.Dropout(p=p_dropout)
        )




#############################
#   STAT-POOLING VARIANTS   #
#############################

class xvector(nn.Module):
    """ Baseline x-vector model using statistics pooling (mean+std) and regular Softmax"""

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

        self.softmax = nn.LogSoftmax()

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
        x = self.softmax(x)

        return x




class xvector_ams(nn.Module):
    """ X-vector model using statistics pooling (mean+std) and Additive Margin Softmax """

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

        self.softmax = nn.LogSoftmax()

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
        x = self.softmax(x)

        return x




##################################
#   ATTENTION POOLING VARIANTS   #
##################################

class xvector_mha(nn.Module):
    """ X-Vector model using Multihead Attention Pooling and regular Softmax """

    def __init__(self, numSpkrs, num_attn_heads, p_dropout=0, rank=0):
        super().__init__()
        self.rank = rank
        
        self.tdnn1 = tdnn_layer(in_channels=30, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=1)
        self.tdnn2 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=2)
        self.tdnn3 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=7, dilation=3)
        self.tdnn4 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=1, dilation=1)
        self.tdnn5 = tdnn_layer(in_channels=512, out_channels=1500, p_dropout=p_dropout, kernel_size=1, dilation=1)

        self.attn_input_size = 1500
        self.mh_attn = MultiHeadAttention(
            key_size=self.attn_input_size, query_size=self.attn_input_size, value_size=self.attn_input_size,
            num_hiddens=1500, num_heads=num_attn_heads
        )

        self.fc1 = fc_embedding_layer(in_channels=1500, out_channels=512, p_dropout=p_dropout)
        self.fc2 = fc_embedding_layer(in_channels=512, out_channels=512, p_dropout=p_dropout)
        self.fc3 = nn.Linear(512,numSpkrs)

        self.softmax = nn.LogSoftmax()

    def forward(self, x, eps=1e-5):
        """Note: x must be (batch_size, feat_dim, chunk_len)"""

        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # add noise
        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape).to(self.rank)
            torch.randn(shape, out=noise)
            x += noise*eps

        # apply Multihead Attention Pooling
        x = torch.moveaxis(x, 1, 2)
        x = self.mh_attn(x, x, x)
        x = torch.sum(x, dim=1)  # compute sum of weighted values per frame
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x



class xvector_mha_ams(nn.Module):
    """ X-Vector model using Multihead Attention Pooling and Additive Margin Softmax """

    def __init__(self, numSpkrs, num_attn_heads, p_dropout=0, rank=0):
        super().__init__()
        self.rank = rank
        
        self.tdnn1 = tdnn_layer(in_channels=30, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=1)
        self.tdnn2 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=5, dilation=2)
        self.tdnn3 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=7, dilation=3)
        self.tdnn4 = tdnn_layer(in_channels=512, out_channels=512, p_dropout=p_dropout, kernel_size=1, dilation=1)
        self.tdnn5 = tdnn_layer(in_channels=512, out_channels=1500, p_dropout=p_dropout, kernel_size=1, dilation=1)

        self.attn_input_size = 1500
        self.mh_attn = MultiHeadAttention(
            key_size=self.attn_input_size, query_size=self.attn_input_size, value_size=self.attn_input_size,
            num_hiddens=1500, num_heads=num_attn_heads
        )

        self.fc1 = fc_embedding_layer(in_channels=1500, out_channels=512, p_dropout=p_dropout)
        self.fc2 = fc_embedding_layer(in_channels=512, out_channels=512, p_dropout=p_dropout)
        self.fc3 = nn.Linear(512,numSpkrs)

        self.softmax = nn.LogSoftmax()

    def forward(self, x, eps=1e-5):
        """Note: x must be (batch_size, feat_dim, chunk_len)"""

        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # add noise
        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape).to(self.rank)
            torch.randn(shape, out=noise)
            x += noise*eps

        # apply Multihead Attention Pooling
        x = torch.moveaxis(x, 1, 2)
        x = self.mh_attn(x, x, x)
        x = torch.sum(x, dim=1)  # compute sum of weighted values per frame
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x






###########################################
###########################################
###########################################
class xvector_legacy(nn.Module):
    """
    DO NOT USE - LEGACY MODEL WITHOUT REFACTORED NN.SEQUENTIAL LAYERS
        (only for extracting xvectors from a saved model with the below structure)
    Baseline x-vector model using statistics pooling (mean+std)
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
