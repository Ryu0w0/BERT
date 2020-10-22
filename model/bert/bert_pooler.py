"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
from torch import nn


class BertPooler(nn.Module):
    """
    It converts features in [CLS]
    """
    def __init__(self, config):
        super(BertPooler, self).__init__()

        # fc layer
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # act_f
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # get feature from the 1st token([CLS])
        first_token_tensor = hidden_states[:, 0]
        # apply fc
        pooled_output = self.dense(first_token_tensor)
        # act_f of Tahn_function
        pooled_output = self.activation(pooled_output)

        return pooled_output
