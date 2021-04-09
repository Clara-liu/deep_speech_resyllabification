'''
available at https://github.com/liuhu-bigeye/enctc.crnn/blob/master/utils.py
'''

from torch.autograd import Variable
import torch


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def WER(preds, targets, blank = 0):
    argmax_preds = torch.argmax(preds, dim=-1)
    n_seq = preds.size()[0]
    n_wrong = 0
    predicted = []
    for n in range(n_seq):
        seq = argmax_preds[n].numpy()
        n_frame = len(seq)
        seq_pred = []
        for t in range(n_frame):
            current_pred = seq[t]
            if t == 0:
                seq_pred.append(current_pred)
            else:
                last_pred = seq_pred[-1]
                if current_pred != last_pred:
                    seq_pred.append(current_pred)
        seq_pred = list(filter(lambda a: a != blank, seq_pred))
        predicted.append(seq_pred)
        target = targets[n].tolist()
        if seq_pred != target:
            n_wrong += 1
    return n_wrong/n_seq, predicted