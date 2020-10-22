import math
import torch

from common.logger import util as com_util
from common.logger.util import logger_


def set_requires_grad(model, p):
    """Specify trainable parameters"""
    # Put False into all of the parameters first
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Put True according to a parameter setting file
    if p.trainable_block_from == "embeddings":
        # all of the parameters are trainable if -1
        for name, param in model.named_parameters():
            param.requires_grad = True
    elif p.trainable_block_from == "encoder":
        for i in range(p.num_trainable_enc_block):
            idx = i + 1
            for name, param in model.bert.encoder.layer[-idx].named_parameters():
                param.requires_grad = True
        for name, param in model.bert.pooler.named_parameters():
            param.requires_grad = True
        for name, param in model.cls.named_parameters():
            param.requires_grad = True
    else:
        assert False, f"Invalid argument: {p.trainable_block_from}"


def disable_all_requires_grad(model):
    # Put False into all of the parameters first
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def logging_model_structure(model):
    com_util.logger_.info("*** MODEL ***")
    com_util.logger_.info(f"{model}")
    com_util.logger_.info(f"*** MODEL'S REQUIRES GRAD ***")
    for name, param in model.named_parameters():
        com_util.logger_.info(f"  name: {name}, grad: {param.requires_grad}")


def load_pretrained_weights(model, load_path):
    # Load weights
    loaded_state_dict = torch.load(load_path)

    # Obtain param name list from the current model
    model.eval()
    param_names = []
    for name, param in model.named_parameters():
        param_names.append(name)

    # Create copied current model weights (and their names)
    new_state_dict = model.state_dict().copy()

    # Setup pre-trained weights using names of copied models
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        if index == 0:
            # skip bert.embeddings.position_ids
            break
        name = param_names[index-1]  # get i-th param name of current model
        new_state_dict[name] = value  # update param in copied current model
        logger_.info(str(key_name)+"->"+str(name))  # logging

        # Escape from loop when finish loading parameters of the current model
        if (index+1 - len(param_names)) >= 0:
            break

    # Load parameters
    model.load_state_dict(new_state_dict)

    return model


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
