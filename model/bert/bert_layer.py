"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
from torch import nn

from model.bert.bert_attention import BertAttention
from model.bert.bert_intermediate import BertIntermediate
from model.bert.bert_output import BertOutput


class BertLayer(nn.Module):
    """  BertLayer plays role of transformer """
    def __init__(self, config):
        super(BertLayer, self).__init__()
        # Self-attention
        self.attention = BertAttention(config)
        # Fully connected layer
        self.intermediate = BertIntermediate(config)
        # Residual connection
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        if attention_show_flg:
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs
        else:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output  # [batch_size, seq_length, hidden_size]

