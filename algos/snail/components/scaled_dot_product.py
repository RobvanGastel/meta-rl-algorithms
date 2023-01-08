import math

import torch
from torch import nn
from torch.functional import F


class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # The attention mechanism
        # Q, K  shape:   [batch, timestep, d_k]
        # V     shape:   [batch, timestep, d_v]

        # 1) MatMul QK^T shape:     [timestep, timestep]
        # All possible pairs of queries and keys
        K_T = K.transpose(-2, -1)
        attention_logits = torch.matmul(Q, K_T)

        # 2) Scale by QK^T/sqrt(d_k)
        # Scale to reduce compute, leads to regions with small gradients
        # in the softmax function
        d_k = Q.size()[-1]
        attention_logits = torch.divide(attention_logits, math.sqrt(d_k))

        # 3) Optional Masking
        # Mask for different sequences lengths in the batch
        if mask:
            attention_logits = attention_logits.masked_fill(mask == 0, -9e15)

        # 4) Softmax(QK^T/sqrt(d_k))
        # Each row represents the attention logits for specific elements i to
        # all other element sin the sequence.
        attention = F.softmax(attention_logits, dim=-1)

        # 5) Softmax(QK^T/sqrt(d_k))V
        # Multiply to obtain a weighted mean, the weights being determined
        # by the attention.
        values = torch.matmul(attention, V)
        return values, attention
