import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_state, model_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % hidden_state == 0

    def forward(self, x):
        pass
