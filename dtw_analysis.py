import librosa
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn



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
    rad = (sound_file_instance.duration_dict['slow'] - sound_file_instance.duration_dict['resyllabified'])/\
          sound_file_instance.duration_dict['resyllabified']*0.5
    for token_row in token_list:
        for token_col in token_list:
            D, wp = librosa.sequence.dtw(sound_file_instance.mel_dict[token_row],
                                         sound_file_instance.mel_dict[token_col],
                                         global_constraints=True,
                                         band_rad=rad,
                                         metric=dist_func)
            wp_dist_list = [D[x, y] for x, y in wp]
            total_dist = sum(wp_dist_list)
            dist_matrix[token_list.index(token_row), token_list.index(token_col)] = total_dist
    return dist_matrix