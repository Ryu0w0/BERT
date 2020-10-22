"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
from torch import nn

from model.bert.bert_layer_norm import BertLayerNorm
from common.util import config as cfg

class BertOutput(nn.Module):
    """ FeedForward block in transformer """
    def __init__(self, config):
        super(BertOutput, self).__init__()

        # fc layer
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        hidden_states： Output from BertIntermediate
        input_tensor：Output from BertAttention
        """
        hidden_states = self.dense(hidden_states)
        cfg.seed_everything()
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
