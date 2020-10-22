import torch
from torch import nn
from common.util import config as cfg


class BertClassifier(nn.Module):
    """Classifier module on BertLayer"""
    def __init__(self, p_model):
        super(BertClassifier, self).__init__()
        self.use_token_cnt_fc = p_model.use_token_cnt_fc if "use_token_cnt_fc" in p_model.keys() else False

        # original layer
        cfg.seed_everything()
        self.cls = nn.Linear(in_features=p_model.hidden_size, out_features=p_model.num_class)
        cfg.seed_everything()
        nn.init.normal_(self.cls.weight, std=0.02)
        cfg.seed_everything()
        nn.init.normal_(self.cls.bias, 0)

        # additional layer
        if self.use_token_cnt_fc:
            self.token_count_mean = 125.33427716479422
            self.token_count_std = 114.56433973046047
            self.token_count_max = 1058
            cfg.seed_everything()
            self.fc_token_len = nn.Linear(in_features=1, out_features=p_model.hidden_size)
            cfg.seed_everything()

    def forward(self, vec_0, token_lens):
        if self.use_token_cnt_fc:
            # normalize
            token_lens = (token_lens - self.token_count_mean) / self.token_count_std
            # reshape as batch
            token_lens = torch.reshape(token_lens, (len(token_lens), 1))
            # fc for token length
            output_length = self.fc_token_len(token_lens)
            # add result
            vec_0 += output_length

        out = self.cls(vec_0)

        return out
