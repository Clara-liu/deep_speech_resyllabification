from torch.autograd import Variable
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from data_generation import LabelConvert

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


def WER(preds: 'network prediction tensor (n batch, time, n class)', targets: 'target labels (n batch, n target)',
        blank=0) -> 'wer and predicted targets':
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


def decode(label_list: 'list of labels') -> 'list of words':
    if not isinstance(label_list, list):
        label_list = label_list.tolist()
    decoded = []
    decoder = LabelConvert()
    for seq in label_list:
        words = ' '.join(decoder.labels_to_words(seq))
        decoded.append(words)
    return decoded

def confusion_collapsed(ypred: 'prediction matrix made by model', ytrue: 'list target labels'):
    converter = LabelConvert()
    predictions = converter.collapsed_to_words(torch.argmax(ypred, 1))
    targets = converter.collapsed_to_words(converter.collapse_seqs(ytrue))
    confuse = confusion_matrix(targets, predictions)
    confuse = pd.DataFrame(confuse, columns=np.unique(predictions), index=np.unique(targets))
    confuse.index.name = 'Actual'
    confuse.columns.name = 'Predicted'
    plt.figure()
    sn.set(font_scale=1.4)
    plot = sn.heatmap(confuse, cmap='Blues', cbar=False, annot=True, annot_kws={'size': 15})
    plot.set_yticklabels(plot.get_yticklabels(), rotation=30)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=30)