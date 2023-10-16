import torch.nn as nn
import numpy as np


class Attention(nn.Module):
    def __init__(self, input_dim, mask):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.mask = mask
        self.softmax = nn.Softmax(dim=2)
        self.q = nn.Linear(input_dim, input_dim)
        self.k = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = q @ k.T / np.sqrt(q.shape[-1]) + self.mask
        attention = self.softmax(scores)
        weighted = attention @ v
        return weighted
