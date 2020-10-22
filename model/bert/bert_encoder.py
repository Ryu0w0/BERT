"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
from torch import nn

from model.bert.bert_layer import BertLayer


class BertEncoder(nn.Module):
    """ A list of BertLayers """
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        all_encoder_layers = []

        # Repeat processing of BertLayer
        for layer_module in self.layer:

            if attention_show_flg:
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
            else:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg)

            # Store attentions for visualization
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # Store attentions for visualization
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        # return attention_probs (last layer) if attention_show is true
        if attention_show_flg:
            return all_encoder_layers, attention_probs
        else:
            return all_encoder_layers
