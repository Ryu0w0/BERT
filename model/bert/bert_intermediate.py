"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
from torch import nn
from model.util import util


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        # FC layer
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # activation func
        self.intermediate_act_fn = util.gelu

    def forward(self, hidden_states):
        """
        hidden_states： output from BertAttention
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
