from torch import nn, from_numpy
import torchaudio
from random import choice
from librosa import power_to_db
import numpy as np
import matplotlib.pyplot as plt

class LabelConvert:
    def __init__(self):
        label_map = '''
        1 bar
        2 skill
        3 skull
        4 bask
        5 ill
        6 Earl
        7 Lee
        8 steal
        9 stale
        10 span
        11 in
        12 least
        13 an
        14 eel
        15 cusp
        16 ale
        17 Kerr
        18 spin
        '''
        # dictionaries to store label to word mapping
        self.syl_map = {}
        self.label_map = {}
        for pair in label_map.strip().split('\n'):
            l, w = pair.split()
            self.syl_map[w] = int(l)
            self.label_map[l] = w
    def words_to_labels(self, words: 'list words in the utterance')-> 'list of int labels':
        label_seq = []
        for w in words:
            label_seq.append(self.syl_map[w])
        return label_seq
    def labels_to_words(self, labels: 'list labels of the words')-> 'list of word strings':
        word_seq = []
        for l in labels:
            word_seq.append(self.label_map[str(l)])
        return word_seq

def converter(wav, sr, nmels = 128, tweak = False, verbose = False):
    win_len = int(sr*0.025)
    hop = int(0.005*sr)
    tweaker = choice(['mask_time', 'mask_frequency'])
    layers = [torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=nmels, win_length=win_len, n_fft=win_len,
                                                   hop_length=hop)]
    if tweak:
        if tweaker == 'mask_time':
            layers.append(torchaudio.transforms.TimeMasking(time_mask_param=15))
        else:
            layers.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=15))
        if verbose:
            print(f'tweak: {tweaker}')
    net = nn.Sequential(*layers)
    converted = from_numpy(power_to_db(net(wav)[0]))

    return converted



def checkMel():
    filename = "sample1.wav"
    waveform, sampling_rate = torchaudio.load(filename)
    original = converter(waveform, sr=sampling_rate)
    tweaked = converter(waveform, sr = sampling_rate, tweak=True, verbose=True)
    fig, ax = plt.subplots(2)
    ax[0].imshow(original, aspect='auto', origin = 'lower')
    ax[1].imshow(tweaked, aspect='auto', origin = 'lower')