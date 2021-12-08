from torch.autograd import Variable
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from data_generation import LabelConvert
from os import listdir

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
    argmax_preds = torch.argmax(preds, dim=-1).cpu()
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

def confusion_collapsed(ypred: 'prediction matrix made by model',
                        ytrue: 'list target labels',
                        speaker_folder: 'path to save the confusion matrix',
                        matrix_label: 'str validation or norm'):
    converter = LabelConvert()
    nlabels = len(converter.seq_dict)
    labels = [converter.seq_dict[x] for x in range(nlabels)]
    predictions = torch.argmax(ypred, 1).tolist()
    targets = converter.collapse_seqs(ytrue).tolist()
    confuse = np.zeros((nlabels, nlabels))
    for i in range(len(targets)):
        confuse[targets[i], predictions[i]] += 1
    confuse = pd.DataFrame(confuse, columns=labels, index=labels)
    # save validation confusion matrix
    confuse.to_csv(f'{speaker_folder}/{matrix_label}_matrix.txt', sep='\t', index=True)
    confuse.index.name = 'Actual'
    confuse.columns.name = 'Predicted'
    plt.figure(figsize=(12, 10))
    sn.set(font_scale=1.4)
    plot = sn.heatmap(confuse, cmap='Blues', cbar=False, annot=True, annot_kws={'size': 15})
    plot.set_yticklabels(plot.get_yticklabels(), rotation=30)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=30)



def transfer_param(net_new: 'new model', path_trained: 'path to pretrained dict'):
    model_dict = net_new.state_dict()
    pretrained_dict = torch.load(path_trained)
    if not isinstance(pretrained_dict, dict):
        pretrained_dict = pretrained_dict.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
    model_dict.update(pretrained_dict)
    net_new.load_state_dict(model_dict)
    return net_new


def early_stopping(eval_metric_log, stop_threshold=0.98):
    if len(eval_metric_log) < 6:
        stop = False
    else:
        past_n_mean = sum(eval_metric_log[-5:])/5
        if past_n_mean >= stop_threshold:
            stop = True
        else:
            stop = False
    return stop

def count_resyllabified(speakers: 'list', folder = 'pilot_2', plot = True, return_data=False):
    # create data frame for counting
    coda_words = ['least_eel', 'least_ale', 'cusp_eel', 'cusp_ale', 'doom_art', 'doom_eat', 'coop_art', 'coop_eat']
    data = {'Speaker': [], 'Word': [], 'Pair': [], 'Count': [0 for i in range(8*len(speakers))]}
    for s in speakers:
        data['Speaker'] = data['Speaker'] + [s for i in range(8)]
        data['Word'] = data['Word'] + coda_words
        data['Pair'] = data['Pair'] + [0, 0, 1, 1, 2, 2, 3, 3]
    data = pd.DataFrame.from_dict(data)
    # loop through each speaker's files and count
    for ss in speakers:
        # get files for each speaker
        path = f'{folder}/{ss}/normal_sound_files/resyllabified'
        files = listdir(path)
        # count
        for f in files:
            # get words
            words = '_'.join(f.split('_')[0:2])
            # find the row idx in predefined dataframe
            condition = (data['Speaker'] == ss) & (data['Word'] == words)
            idx = data.index[condition]
            data.loc[idx, 'Count'] += 1
    if plot:
        grid = sn.FacetGrid(data, row='Speaker')
        grid.map_dataframe(sn.barplot, x='Word', y='Count', hue='Pair', palette='Set2')
        grid.add_legend()
    if return_data:
        return data


def sum_plot_confusion(speakers: 'list'):
    # initialise zero matrices
    summed_norm_data = np.zeros((12, 12))
    summed_validation_data = np.zeros((12, 12))
    for s in speakers:
        # path to validation and normal speaking rate matrices
        norm_path = f'pilot_2/{s}/norm_matrix.txt'
        validation_path = f'pilot_2/{s}/validation_matrix.txt'
        # read the data for the metrices
        norm_data = pd.read_csv(norm_path, sep='\t', index_col=0)
        validation_data = pd.read_csv(validation_path, sep='\t', index_col=0)
        # filter out P1 rows
        unwanted = ['Kerr speel', 'Kerr spale', 'cusp eel', 'cusp ale']
        norm_data = norm_data.drop(unwanted, axis=0)
        validation_data = validation_data.drop(unwanted, axis=0)
        # record word labels
        if speakers.index(s) == 0:
            labels = list(norm_data.index)
        # filter out P1 cols then convert to numpy
        norm_data = norm_data.drop(unwanted, axis=1).to_numpy()
        validation_data = validation_data.drop(unwanted, axis=1).to_numpy()
        # sum matrices
        summed_norm_data += norm_data
        summed_validation_data += validation_data
    # convert to pandas
    summed_norm_data = pd.DataFrame(summed_norm_data, columns=labels, index=labels)
    summed_validation_data = pd.DataFrame(summed_validation_data, columns=labels, index=labels)
    # add col and row titles for the heatmap
    summed_norm_data.index.name = 'Actual'
    summed_norm_data.columns.name = 'Predicted'
    summed_validation_data.index.name = 'Actual'
    summed_validation_data.columns.name = 'Predicted'
    # plot and save heatmaps
    plt.figure(figsize=(12, 10))
    sn.set(font_scale=1.4)
    plot = sn.heatmap(summed_norm_data, cmap='Blues', cbar=False, annot=True, annot_kws={'size': 15})
    plot.set_yticklabels(plot.get_yticklabels(), rotation=30)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
    plt.savefig(f'pilot_2/summed_norm.jpg')
    plt.clf()
    plot = sn.heatmap(summed_validation_data, cmap='Blues', cbar=False, annot=True, annot_kws={'size': 15})
    plot.set_yticklabels(plot.get_yticklabels(), rotation=30)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
    plt.savefig(f'pilot_2/summed_validation.jpg')
    plt.close()







