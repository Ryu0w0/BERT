import logging
import time

import numpy as np

from logging import getLogger, StreamHandler, Formatter, FileHandler
from contextlib import contextmanager
from sklearn import metrics


logger_ = None
writer_ = None
collector_ = None


class StatCollector:
    def __init__(self, path_dict, args, writer_):
        self.save_root = path_dict.save_root
        self.save_key = args.save_key
        self.writer_ = writer_
        # contain for a single epoch
        self.train_y_true_tmp = []
        self.train_y_pred_tmp = []
        self.val_y_true_tmp = []
        self.val_y_pred_tmp = []
        self.train_loss_tmp = []
        self.val_loss_tmp = []
        self.train_time_tmp = []
        self.val_time_tmp = []
        # contain all epochs
        self.train_recall = []
        self.train_prec = []
        self.train_acc = []
        self.val_recall = []
        self.val_prec = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.train_time = []
        self.val_time = []

    def initialize_per_epoch(self):
        self.train_y_true_tmp = []
        self.train_y_pred_tmp = []
        self.val_y_true_tmp = []
        self.val_y_pred_tmp = []
        self.train_loss_tmp = []
        self.val_loss_tmp = []
        self.train_time_tmp = []
        self.val_time_tmp = []

    def finalize_stat_per_epoch(self, epoch):
        # for acc, recall, and precision
        for phase in ["train", "val"]:
            y_true, y_pred = eval(f"self.{phase}_y_true_tmp"), eval(f"self.{phase}_y_pred_tmp")
            recall = metrics.recall_score(y_true, y_pred, average=None)
            precision = metrics.precision_score(y_true, y_pred, average=None)
            acc = metrics.accuracy_score(y_true, y_pred)
            eval(f"self.{phase}_recall").append(recall)
            eval(f"self.{phase}_prec").append(precision)
            eval(f"self.{phase}_acc").append(acc)
            tag = f"{phase}_epoch/acc"
            # only record acc
            self.writer_.add_scalar(tag=f"{tag}", scalar_value=acc, global_step=epoch + 1)

        for phase in ["train", "val"]:
            for name in ["loss", "time"]:
                avg = np.mean(eval(f"self.{phase}_{name}_tmp"))
                eval(f"self.{phase}_{name}").append(avg)
                tag = f"{phase}_epoch/{name}"
                self.writer_.add_scalar(tag=f"{tag}", scalar_value=avg, global_step=epoch + 1)

        self.initialize_per_epoch()

    def record_lr(self, optimizer, step):
        """ Contain learning rate every step """
        for idx, group in enumerate(optimizer.param_groups):
            updated_lr = group["lr"]
            self.writer_.add_scalar(tag=f"train_step/group{idx}", scalar_value=updated_lr, global_step=step)

    def set_stat_per_batch(self, phase, stat_name, stat):
        """ Run every batch """
        if isinstance(stat, list):
            eval(f"self.{phase}_{stat_name}_tmp").extend(stat)
        else:
            eval(f"self.{phase}_{stat_name}_tmp").append(stat)

    def report_cur_lr(self, optimizer):
        """ Contain learning rate every step """
        for idx, group in enumerate(optimizer.param_groups):
            updated_lr = group["lr"]
            logger_.info(f"[Learning Rate] group{idx}: {updated_lr}")

    def report_by_epoch(self, epoch):
        for phase in ["train", "val"]:
            for name in ["acc", "recall", "prec", "loss", "time"]:
                logger_.info(f"[epoch {epoch + 1} {phase}] {name}: {np.around(eval(f'self.{phase}_{name}')[epoch], 4)}")

    def report_at_last(self):
        for phase in ["train", "val"]:
            for name in ["acc", "loss", "time"]:
                target = eval(f'self.{phase}_{name}')
                max_, avg_ = np.max(target), np.mean(target)
                max_, avg_ = np.around(max_, 4), np.around(avg_, 4)
                logger_.info(f"[Summary {phase}] {name}: max {max_}, avg {avg_}")


@contextmanager
def timer(name, logger_=None, level=logging.DEBUG,
          writer=None, writer_tag=None, writer_gb_step=None,
          collector=None, col_phase=None):
    """
    Logging the time count.
        name: str
            header of log
        logger_: instance of logger
            supposed to set logger only for time count to separate log file
    """
    print_ = print if logger_ is None else lambda msg: logger_.log(level, msg)
    time_st = time.time()
    print_(f'[{name}] start')
    yield
    cost = time.time() - time_st
    print_(f'[{name}] done in {cost:.0f} s')
    if writer:
        writer.add_scalar(tag=writer_tag, scalar_value=cost, global_step=writer_gb_step)
    if collector:
        collector.set_stat_per_batch(col_phase, "time", cost)


def create_logger(logger_name, save_root, save_key, level=logging.INFO):
    # logger object
    logger = getLogger(f"{logger_name}")

    # set debug log level, level is controlled in handler
    logger.setLevel(logging.DEBUG)

    # create log handler
    stream_handler = StreamHandler()
    file_handler = FileHandler(f'{save_root}/logs/{save_key}.log', 'a')

    # set log level
    stream_handler.setLevel(level)
    file_handler.setLevel(level)

    # set format
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)
    file_handler.setFormatter(handler_format)

    # set handler to logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_log_level_from_name(log_level_name):
    if log_level_name == "DEBUG":
        return logging.DEBUG
    elif log_level_name == "INFO":
        return logging.INFO
    else:
        assert False, f"Log level is either INFO or DEBUG"


def report_dataset_size(train_ds, val_ds):
    train_count = {"0": 0, "1": 0}
    valid_count = {"0": 0, "1": 0}
    train_num_gt_256 = {"0": 0, "1": 0}
    valid_num_gt_256 = {"0": 0, "1": 0}
    for e in train_ds.examples:
        train_count[e.Label] += 1
        train_num_gt_256[e.Label] += 1 if len(e.Text) > 256 else 0
    for e in val_ds.examples:
        valid_count[e.Label] += 1
        valid_num_gt_256[e.Label] += 1 if len(e.Text) > 256 else 0
    logger_.info(f"train num: {len(train_ds)}, neg_num: {train_count['0']}, pos_num: {train_count['1']}, "
                 f"neg_gt_256: {train_num_gt_256['0']}, pos_gt_256: {train_num_gt_256['1']}")
    logger_.info(f"valid num: {len(val_ds)}, neg_num: {valid_count['0']}, pos_num: {valid_count['1']}, "
                 f"neg_gt_256: {valid_num_gt_256['0']}, pos_gt_256: {valid_num_gt_256['1']}")
