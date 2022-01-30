"""
Train and eval functions used in main.py
"""
import logging
import paddle
from pathlib import Path


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, output_dir, args):
    output_dir = Path(output_dir)
    model.train()
    for batch_id, data in enumerate(data_loader):
        samples, targets = data[0], data[1]
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        acc = paddle.metric.accuracy(outputs, targets.unsqueeze(-1))
        if batch_id % 200 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            logging.info("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
        if batch_id % 2000 == 0:
        # if batch_id % 10 == 0:
            checkpoint_path = output_dir / 'checkpoint_epoch_{}_batchid_{}_acc_{}.pdparams'.format(epoch, batch_id, acc.item())
            paddle.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, str(checkpoint_path))