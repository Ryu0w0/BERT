"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
import torch
from torch import nn


class BertLayerNorm(nn.Module):
    """ LayerNormalization """
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weight
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # bias
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
