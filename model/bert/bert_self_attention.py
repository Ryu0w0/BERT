"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
import math

import torch
from torch import nn

from common.util import config as cfg


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size

        # fc layers calculating attention weights
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # weights are applied into value
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        Transpose a tensor into a shape for multi-head attention
            [batch_size, seq_len, hidden] → [batch_size, num_heads, seq_len, hidden/num_heads]
        """
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        # prepare query, key and value
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # change shape of query, key and value into multi-head-shape
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # calculate attention weights
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        # apply mask
        attention_scores = attention_scores + attention_mask

        # normalize attention weights by softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # drop out
        cfg.seed_everything()
        attention_probs = self.dropout(attention_probs)

        # apply attention weights into values
        context_layer = torch.matmul(attention_probs, value_layer)

        # convert back into concatinated multi-heads
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # return attention probs for visualization purpose
        if attention_show_flg:
            return context_layer, attention_probs
        else:
            return context_layer
