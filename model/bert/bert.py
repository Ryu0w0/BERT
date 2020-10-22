"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
import torch
from torch import nn

from model.bert.bert_embeddings import BertEmbeddings
from model.bert.bert_encoder import BertEncoder
from model.bert.bert_pooler import BertPooler


class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        """
        input_ids： [batch_size, sequence_length] vocab id list
        token_type_ids： [batch_size, sequence_length] id of order of sentence
        attention_mask：mask same with transformer
        output_all_encoded_layers：whether return all attention layers or only the last layer
        attention_show_flg：whether return weights of Self-Attention or not, visualization purpose
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # convert mask into [minibatch, 1, 1, seq_length] to use in multi-head Attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # convert values of masks from 0 or 1 into 0 and -inf such that -inf makes value zero in softmax
        # here use -10000 instead of -inf
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # BertEmbedding
        # convert ids into embedding
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # A list of BertLayers
        if attention_show_flg:
            '''return attention_probs when attention_show is true '''
            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)
        else:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)
        # BertPooler
        # Feed features from the last layer
        pooled_output = self.pooler(encoded_layers[-1])

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # also return attention_probs if attention_show is true
        if attention_show_flg:
            return encoded_layers, pooled_output, attention_probs
        else:
            return encoded_layers, pooled_output
