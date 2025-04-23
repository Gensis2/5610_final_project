import torch
import pandas as pd

def adjust_learning_rate(optimizer, cur_epoch, max_epoch):
    if cur_epoch == (max_epoch*0.5) or cur_epoch == (max_epoch*0.7) or cur_epoch==(max_epoch*0.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10


def accuracy(outp, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = outp.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def combine_snn_df():
    df = pd.DataFrame(columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_acc'])
    for batch_size in ['32', '64', '128', '256']:
        for lr in ['0.2', '0.1', '0.05', '0.01']:
            for num_epochs in ['5', '10', '25', '50']:
                path = f'batch_size/{batch_size}/lr/{lr}/num_epochs/{num_epochs}/snn_results.csv'
                epoch_df = pd.read_csv(path)
                df = pd.concat([df, epoch_df], ignore_index=True)
                df = df.drop_duplicates()  # Remove duplicate rows

    df.to_csv('data_processing/snn_results.csv', index=False)