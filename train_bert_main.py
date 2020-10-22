def initialization():
    """
    1. Process arguments
    2. Create logger
    3. Create dictionary of paths
    Return arguments, path_dict.
    """
    import argparse
    from attrdict import AttrDict
    from common.logger import util as log_util
    from torch.utils.tensorboard import SummaryWriter
    import common.util.util as com_util
    from common.util import config as cfg

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_key", type=str, default="bert")
    parser.add_argument("-load_key_weight", type=str, default="pytorch_model",
                        help="Used to load pre-trained weights. File name should be **.bin")
    parser.add_argument("-load_key_param", type=str, default="small_additional_fc_512")
    parser.add_argument("-batch_size_train", type=int, default=4)
    parser.add_argument("-batch_size_valid", type=int, default=4)
    parser.add_argument("-epoch", type=int, default=3)
    parser.add_argument("-feed_seed", type=int, default=1)
    parser.add_argument("-log_level", type=str, default="INFO")
    parser.add_argument("-train_file_name", type=str, default="yelp_train.tsv")
    args = parser.parse_args()

    # Update config
    cfg.feed_seed = False if args.feed_seed == 0 else True

    # variables related to path
    path_dict = {
        "load_root": "./files/load",
        "save_root": "./files/output",
        "input_root": "./files/input",
        "data_name": "yelp",
        "vocab_name": "vocab.txt",
        "train_file_name": f"{args.train_file_name}",
        "save_key": args.save_key
    }
    path_dict = AttrDict(path_dict)

    # model hyper-parameters from file
    p = com_util.get_json_parameters("./model/params", args.load_key_param)
    p = AttrDict(p)

    # create logger
    args.log_level = log_util.get_log_level_from_name(args.log_level)
    logger_ = log_util.create_logger("main", path_dict.save_root, args.save_key, args.log_level)
    log_util.logger_ = logger_

    # create tensor-board writer
    writer_ = SummaryWriter(log_dir=f"{path_dict.save_root}/board/{args.save_key}")
    log_util.writer_ = writer_

    # create statistics collector
    collector_ = log_util.StatCollector(path_dict, args, writer_)
    log_util.collector_ = collector_

    # logging
    logger_.info("*** ARGUMENTS ***")
    logger_.info(args)
    logger_.info("*** PARAMETERS ***")
    logger_.info(p)

    return args, path_dict, p


def main():
    # it has to be run at the beginning of main()
    args, path_dict, p = initialization()

    import random

    import numpy as np
    import torch
    from torch import nn
    import torch.optim as optim
    import torchtext
    from transformers.optimization import get_linear_schedule_with_warmup

    from common.tokenizer import builder as tk_builder
    from common.logger.util import logger_  # it is logger, it can be used as logger_.info()
    from common.logger import util as log_util
    from common.util import text_processing as txt_util
    from common.util import train
    from common.util import config as cfg
    from model.util import util as m_util
    from model.bert.bert import BertModel
    from model.bert.bert_yelp import BertForYelp

    # Build tokenizer including pre-processing and tokenization
    tokenizer = tk_builder.build_tokenizer(f"{path_dict.load_root}/vocab/vocab.txt")

    # Define data Fields
    logger_.info("** DATASET **")
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True,
                                lower=True, include_lengths=True, batch_first=True, fix_length=p.txt_prc.max_length,
                                init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[unk]')
    LABEL = torchtext.data.LabelField(sequential=False, use_vocab=False)
    # LEN = torchtext.data.LabelField(preprocessing=lambda x: int(x), sequential=False, use_vocab=False)

    # Define dataset
    train_val_ds = torchtext.data.TabularDataset(
        path=f'{path_dict.load_root}/dataset/{path_dict.data_name}/{path_dict.train_file_name}',
        format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])
    train_ds, val_ds = train_val_ds.split(
        split_ratio=0.8, random_state=random.seed(cfg.cur_seed))
    del train_val_ds
    log_util.report_dataset_size(train_ds, val_ds)

    # Construct vocabulary and set it in TEXT referred by dataset
    logger_.info("** VOCAB **")
    vocab_bert, ids_to_tokens_bert = txt_util.load_vocab(
        vocab_file=f"{path_dict.load_root}/vocab/{path_dict.vocab_name}")
    #  temporary build vocab based on words used in train dataset but updated by vocabs from whole words
    TEXT.build_vocab(train_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_bert
    logger_.info(f"Vocab file name: {path_dict.vocab_name}, length: {len(vocab_bert)}")

    # Create Iterator a.k.a DataLoader
    cfg.seed_everything()
    train_dl = torchtext.data.Iterator(
        train_ds, batch_size=args.batch_size_train, train=True, shuffle=True)
    cfg.seed_everything()
    val_dl = torchtext.data.Iterator(
        val_ds, batch_size=args.batch_size_valid, train=False, sort=False, shuffle=True)
    dataloaders_dict = {"train": train_dl, "val": val_dl}

    # Build model
    #  build Bert part according to parameters
    sub_model = BertModel(p.model)
    #  load pre-trained parameters
    sub_model = m_util.load_pretrained_weights(sub_model, f"{path_dict.load_root}/model/{args.load_key_weight}.bin")
    #  build whole model
    model = BertForYelp(sub_model, p.model)
    # set_requires_grad
    m_util.set_requires_grad(model, p.model)
    m_util.logging_model_structure(model)

    # Optimiser
    logger_.info("*** OPTIMIZER ***")
    optimizer = optim.Adam([
        {'params': model.bert.encoder.layer[-1].parameters(), 'lr': p.opt.lr_grp1},
        {'params': model.cls.parameters(), 'lr': p.opt.lr_grp2}
    ], betas=(0.9, 0.999))
    logger_.info(f"{optimizer}")

    # Scheduler
    #  1st epoch for warming up, the rest for decaying
    logger_.info("*** SCHEDULER ***")
    num_warmup_steps = np.ceil(len(train_ds) / args.batch_size_train)
    num_training_steps = args.epoch * num_warmup_steps
    # num_training_steps: whole training steps including warming up
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    logger_.info(f"num_warmup_steps: {num_warmup_steps}, num_training_steps: {num_training_steps}")

    # Loss function
    logger_.info("*** LOSS FUNC ***")
    criterion = nn.CrossEntropyLoss()
    logger_.info(f"{criterion}")

    # Training and validation
    model_trained = train.train_model(model, dataloaders_dict,
                                      criterion, optimizer, scheduler, num_epochs=args.epoch)

    # Save trained parameters
    save_path = f'{path_dict.save_root}/weights/{path_dict.save_key}.ptn'
    torch.save(model_trained.state_dict(), save_path)


if __name__ == '__main__':
    main()


