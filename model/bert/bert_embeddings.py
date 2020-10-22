"""
It refers to https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
"""
import torch
from torch import nn

from common.util import config as cfg
from model.bert.bert_layer_norm import BertLayerNorm


class BertEmbeddings(nn.Module):
    """
    Convert id (vocab id, seq id, sentence id) into embedding
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # Token Embedding：convert vocab id into embedding
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)

        # Transformer Positional Embedding：position id of token into embedding
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        # Sentence Embedding：seq id of sentence (1 or 2) into embedding
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        # 1. Token Embeddings
        words_embeddings = self.word_embeddings(input_ids)

        # 2. Sentence Embedding
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 3. Transformer Positional Embedding：
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Add up 3 embeddings, [batch_size, seq_len, hidden_size]
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # LayerNormalization and Dropout
        embeddings = self.LayerNorm(embeddings)
        cfg.seed_everything()
        embeddings = self.dropout(embeddings)

        return embeddings
