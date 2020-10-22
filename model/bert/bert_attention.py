"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
from torch import nn

from model.bert.bert_self_attention import BertSelfAttention
from model.bert.bert_self_output import BertSelfOutput


class BertAttention(nn.Module):
    """ Self-Attention in BertLayer """
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        if attention_show_flg == True:
            self_output, attention_probs = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output
