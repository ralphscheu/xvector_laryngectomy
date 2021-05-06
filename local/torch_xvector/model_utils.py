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
