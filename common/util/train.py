import torch
import logging

from common.util import config as cfg
from common.logger.util import timer, logger_, writer_, collector_


def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs):

    # Set device and transfer to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger_.info(f"Device：{device}")
    logger_.info("*** TRAINING ***")

    # Enable this option in case it is available
    torch.backends.cudnn.benchmark = True
    train_step = 1
    for epoch in range(num_epochs):
        # specify seed in accordance with epoch
        if cfg.feed_seed:
            cfg.cur_seed = epoch + 1

        for phase in ['train', 'val']:
            with timer(f"{phase}", logger_, logging.INFO, writer_, f"{phase} epoch/time", epoch + 1,
                       collector=collector_, col_phase=phase):
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                for batch in (dataloaders_dict[phase]):
                    reviews = batch.Text[0].to(device)
                    labels = batch.Label.to(device)
                    lens = [len([token for token in tokens if token > 0]) for tokens in reviews.detach().numpy()]
                    lens = torch.tensor(lens).to(device)
                    batch_size_ = len(labels)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(reviews, lens, token_type_ids=None, attention_mask=None,
                                        output_all_encoded_layers=False, attention_show_flg=False)
                        loss = criterion(outputs, labels)
                        # extract predicted class having the highest prob
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            collector_.record_lr(optimizer, train_step)
                            train_step += 1

                        # collect statistics for reporting
                        collector_.set_stat_per_batch(phase, "loss", loss.item() * batch_size_)
                        collector_.set_stat_per_batch(phase, "y_pred", preds.cpu().tolist())
                        collector_.set_stat_per_batch(phase, "y_true", labels.data.cpu().tolist())

        # report statistics per epoch
        collector_.finalize_stat_per_epoch(epoch)
        collector_.report_by_epoch(epoch)
        collector_.report_cur_lr(optimizer)

    # report max and avg statistics among whole epochs
    collector_.report_at_last()

    return model
