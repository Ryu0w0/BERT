import os
import random
import numpy as np
import torch

# GLOBAL VARIABLE
cur_seed = 1  # it links with epoch, and epoch begins from 1, hence 1 is specified here
feed_seed = False


def seed_everything(target="all"):
    if feed_seed:
        # torch is slow
        if target in ["torch", "all"]:
            torch.manual_seed(cur_seed)
            torch.cuda.manual_seed(cur_seed)
            torch.backends.cudnn.deterministic = True

        if target in ["random", "all"]:
            random.seed(cur_seed)
            np.random.seed(cur_seed)
            os.environ['PYTHONHASHSEED'] = str(cur_seed)

