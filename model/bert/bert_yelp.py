from torch import nn
from model.bert.bert_classifier import BertClassifier


class BertForYelp(nn.Module):
    """ Bert with classifier module for Yelp review sentiment analysis """
    def __init__(self, net_bert, p_model):
        super(BertForYelp, self).__init__()
        # BERT
        self.bert = net_bert
        # classifier
        self.cls = BertClassifier(p_model)
        # for logarithmically normalising token count
        self.token_count_mean = 4.493268948472764
        self.token_count_std = 0.8326020319638255
        # for reshape tensor in forward
        self.hidden_size = p_model.hidden_size

    def forward(self, input_ids, token_lens,
                token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, attention_show_flg=False):
        """
        input_ids： sentences of token ids [batch_size, sequence_length]
        token_type_ids： seq_sentence ids (1st or 2nd sentence) [batch_size, sequence_length]
            it is always the same value for this task (fine-tuning)
        output_all_encoded_layers：whether return all of BertLayrers' outputs or only the last one
        attention_show_flg：whether return weights of Self-Attention or not
        """
        if attention_show_flg:
            encoded_layers, pooled_output, attention_probs = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)
        else:
            encoded_layers, pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)

        # classification by features in [CLS]
        vec_0 = encoded_layers[:, 0, :]
        vec_0 = vec_0.view(-1, self.hidden_size)
        out = self.cls(vec_0, token_lens)

        if attention_show_flg:
            return out, attention_probs
        else:
            return out