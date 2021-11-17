import librosa
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from stimuli import code_dict, stimuli_pilot



class sound_files:
    def __init__(self, speaker, paths_dict):
        self.speaker = speaker
        self.paths_dict = paths_dict
        # dict for wav signals
        self.wav_dict = {}
        # sampling rate
        self.sr = 0
        # dict for mel features
        self.mel_dict = {}
        # dict for mean duration of each condition
        self.duration_dict = {condition: [] for condition, _ in self.paths_dict.items()}
    def get_wavs(self):
        for condition, path in self.paths_dict.items():
            current_folder_files = sorted([path + '/' + x for x in listdir(path) if '.wav' in x])
            for f in current_folder_files:
                # create a key for each token indicating sound and condition
                current_token = f.split('.')[0].split('/')[-1] + f'_{condition}'
                sig, sr = librosa.load(f)
                # check if wav_dict is empty
                if not bool(self.wav_dict):
                    self.sr = sr
                self.wav_dict[current_token] = sig
                current_dur = sig.shape[0]/sr
                self.duration_dict[condition].append(current_dur)
        self.duration_dict = {condition: sum(dur_list)/len(dur_list) for condition, dur_list in self.duration_dict.items()}
    def get_mels(self, hop_s, win_s, n_mels):
        hop_int = int(hop_s*self.sr)
        win_int = int(win_s*self.sr)
        for token, sig in self.wav_dict.items():
            self.mel_dict[token] = librosa.feature.melspectrogram(sig, sr=self.sr, n_mels=n_mels,
                                                                  n_fft=win_int, hop_length=hop_int)

def calc_dtw_dist(sound_file_instance: 'instance of the sound_files class',
                  dist_func: 'str function to calculate feature vector distance e.g. cosine')-> 'matrix of distance':
    # get list of token labels
    token_list = list(sound_file_instance.mel_dict.keys())
    n_tokens = len(token_list)
    # initialise distance matrix
    dist_matrix = np.zeros((n_tokens, n_tokens))
    # calculate sako-chiba band radius for restricting warping path
    # rad = (sound_file_instance.duration_dict['slow'] - sound_file_instance.duration_dict['resyllabified'])/\
    #       sound_file_instance.duration_dict['resyllabified']*0.5
    for token_row in token_list:
        for token_col in token_list:
            # row_sig_len = sound_file_instance.mel_dict[token_row].shape[1]
            # col_sig_len = sound_file_instance.mel_dict[token_col].shape[1]
            # # calculate sako-chiba band radius for restricting warping path
            # rad = (abs(row_sig_len - col_sig_len)/min([row_sig_len, col_sig_len]))*0.5
            # if rad < 0.1:
            #     rad = 0.1
            D, wp = librosa.sequence.dtw(sound_file_instance.mel_dict[token_row],
                                         sound_file_instance.mel_dict[token_col],
                                         global_constraints=True,
                                         #band_rad=rad,
                                         metric=dist_func)
            wp_dist_list = [D[x, y] for x, y in wp]
            total_dist = sum(wp_dist_list)
            dist_matrix[token_list.index(token_row), token_list.index(token_col)] = total_dist
    return token_list, dist_matrix


def take_rep_mean(dist_matrix,
                  label_list,
                  reduce_v_contrast = True)-> 'dist matrix averaged between repetition,' \
                                              'if reduce_v_contrast, dist martix further averaged between vowel pairs':
    # reduce repetitions and retain insertion order e.g. ['a', 'a', 'b'] -> ['a', 'b']
    label_list_reduced = list(dict.fromkeys([f'{x[0]}_{x[1]}_{"_".join(x[3:])}' for x in [x_.split('_') for x_ in label_list]]))
    # indices of repeated labels
    rep_indices = []
    for l in label_list_reduced:
        indices = []
        for l_ in label_list:
            # see if the words in the reduced one is a subset of the non reduced ones
            if set(l.split('_')) <= set(l_.split('_')):
                indices.append(label_list.index(l_))
        rep_indices.append(indices)
    rep_indices = [(x[0], x[-1]) for x in rep_indices]
    # to record mean values
    mean_matrix = np.zeros((len(label_list_reduced), len(label_list_reduced)))
    # loop through rep indices to subset dist_matrix and calculate mean
    for row in range(len(rep_indices)):
        for col  in range(len(rep_indices)):
            row_start, row_end = rep_indices[row]
            col_start, col_end = rep_indices[col]
            mean = dist_matrix[row_start:row_end + 1, col_start:col_end + 1].mean()
            mean_matrix[row, col] = mean
    if reduce_v_contrast:
        label_list_reduced_further = list(dict.fromkeys([f'{code_dict[x[0]]}_{"_".join(x[2:])}' for x in [x_.split('_') for x_ in label_list_reduced]]))
        indices_reduced_further = []
        for l in label_list_reduced_further:
            indices = []
            for l_ in label_list_reduced:
                if l == code_dict[l_.split('_')[0]] + '_' + '_'.join(l_.split('_')[2:]):
                    indices.append(label_list_reduced.index(l_))
            indices_reduced_further.append(indices)
        indices_reduced_further = [(x[0], x[-1]) for x in indices_reduced_further]
        further_mean_matrix = np.zeros((len(label_list_reduced_further), len(label_list_reduced_further)))
        for row in range(len(indices_reduced_further)):
            for col  in range(len(indices_reduced_further)):
                row_start, row_end = indices_reduced_further[row]
                col_start, col_end = indices_reduced_further[col]
                mean = mean_matrix[row_start:row_end + 1, col_start:col_end + 1].mean()
                further_mean_matrix[row, col] = mean
        return label_list_reduced_further, np.round(further_mean_matrix, 2)
    else:
        return label_list_reduced, np.round(mean_matrix, 2)


def get_heatmap(speaker: 'str speaker abbreviation',
         reduction_level: 'str none, rep or all')-> 'none output png to speaker folder':
    speaker_folder_path = f'pilot_2/{speaker}'
    path_dict = {'slow': f'pilot_2/{speaker}/slow_sound_files',
                 'non_resyllabified': f'pilot_2/{speaker}/normal_sound_files/non_resyllabified',
                 'resyllabified': f'pilot_2/{speaker}/normal_sound_files/resyllabified'}
    sound_obj = sound_files(speaker, path_dict)
    sound_obj.get_wavs()
    sound_obj.get_mels(0.01, 0.03, 26)
    raw_labels, raw_dist_matrix = calc_dtw_dist(sound_obj, 'cosine')
    if reduction_level == 'none':
        data = pd.DataFrame(raw_dist_matrix, columns=raw_labels, index=raw_labels)
        plot = sn.heatmap(data, cmap="YlGnBu", annot=False, cbar=True)
        fig = plot.get_figure()
        fig.savefig(f'{speaker_folder_path}/{speaker}_dtw_dist.png', dpi = 700)
    else:
        if reduction_level == 'rep':
            reduced_labels, reduced_matrix = take_rep_mean(raw_dist_matrix, raw_labels, reduce_v_contrast=False)
        else:
            reduced_labels, reduced_matrix = take_rep_mean(raw_dist_matrix, raw_labels)
        data = pd.DataFrame(reduced_matrix, columns=reduced_labels, index=reduced_labels)
        plt.figure(figsize=(16, 13))
        sn.set(font_scale=.9)
        plot = sn.heatmap(data, cmap="YlGnBu", linewidths=.5, annot=False, cbar=True, xticklabels=1, yticklabels=1)
        fig = plot.get_figure()
        fig.savefig(f'{speaker_folder_path}/{speaker}_dtw_dist.png', dpi = 700)

def get_dist_data(speaker: 'str speaker abbreviation',
                  reference_condition: 'str resyllabified vs slow or non_resyllabified vs slow'):
    speaker_folder_path = f'pilot_2/{speaker}'
    path_dict = {'slow': f'pilot_2/{speaker}/slow_sound_files',
             'non_resyllabified': f'pilot_2/{speaker}/normal_sound_files/non_resyllabified',
             'resyllabified': f'pilot_2/{speaker}/normal_sound_files/resyllabified'}
    sound_obj = sound_files(speaker, path_dict)
    sound_obj.get_wavs()
    sound_obj.get_mels(0.01, 0.03, 26)
    raw_labels, raw_dist_matrix = calc_dtw_dist(sound_obj, 'cosine')
    # initialise empty list for data. col order - Distance, Pair, Comparison
    data = []
    # using row as the reference sound
    for row in range(len(raw_labels)):
        for col in range(len(raw_labels)):
            row_label_list = raw_labels[row].split('_')
            col_label_list = raw_labels[col].split('_')
            # if the label is resyllabified or non_resyllabified depending on the condition specified
            if '_'.join(row_label_list[3:]) == reference_condition:
                # if the reference and query sounds are from the same word pair
                if (code_dict[row_label_list[0]].split('_')[0]) == (code_dict[col_label_list[0]].split('_')[0]):
                    # if the vowel target matches between the two
                    row_stimuli_idx = stimuli_pilot.index(' '.join(row_label_list[:2]))
                    col_stimuli_idx = stimuli_pilot.index(' '.join(col_label_list[:2]))
                    if row_stimuli_idx - col_stimuli_idx == 2 or row_stimuli_idx == col_stimuli_idx:
                        distance = raw_dist_matrix[row, col]
                        pair_code = code_dict[row_label_list[0]].split('_')[0]
                        comparison = f'{reference_condition}_coda_VS_slow_{code_dict[col_label_list[0]].split("_")[-1]}'
                        data.append([distance, pair_code, comparison, speaker])
    result = pd.DataFrame(data, columns=['Distance', 'Pair', 'Comparison', 'Speaker'])
    return result


def main(reference_condition, speaker_list):
    for s in speaker_list:
        current_data = get_dist_data(s, reference_condition)
        if speaker_list.index(s) == 0:
            data = current_data
        else:
            data = pd.concat([data, current_data])
    data.to_csv(f'pilot_2/dtw_analysis_{reference_condition}_comparison.txt', sep='\t', index=False)

if __name__ == '__main__':
    main('resyllabified', ['BS', 'AR', 'FE', 'GJ', 'MAG', 'RB', 'SG', 'TB'])






