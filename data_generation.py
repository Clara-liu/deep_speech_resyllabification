import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc, logfbank
from random import choice, shuffle
from math import ceil
from librosa import power_to_db
from librosa.util import normalize
from torch import nn, from_numpy, Tensor, utils, rand, mean
from os import listdir
#torchaudio.set_audio_backend('sox_io')

class LabelConvert:
    def __init__(self):
        label_map = '''
        0 Lee
        1 steal
        2 stale
        3 least
        4 eel
        5 ale
        6 Kerr
        7 speel
        8 spale
        9 cusp
        10 do
        11 meet
        12 mart
        13 doom
        14 art
        15 eat
        16 coo
        17 part
        18 Pete
        19 coop
        '''
        self.seq_dict = {0: 'Lee steal', 1: 'Lee stale', 2: 'least eel', 3: 'least ale', 4: 'Kerr speel', 5: 'Kerr spale',
                    6: 'cusp eel', 7: 'cusp ale', 8: 'do mart', 9: 'do meet', 10: 'doom art', 11: 'doom eat',
                    12: 'coo part', 13: 'coo Pete', 14: 'coop art', 15: 'coop eat'}
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
    def collapsed_to_words(self, collasped_labels: 'list collapsed labels')-> 'list word sequences':
        if not isinstance(collasped_labels, list):
            collasped_labels = collasped_labels.tolist()
        words_seq = []
        for label in collasped_labels:
            words_seq.append(self.seq_dict[label])
        return words_seq
    def collapse_seqs(self, labels: 'list lists of labels of the words')->'list containing one int for seq class':
        collapsed = []
        for l in labels:
            if 0 in l:
                if 1 in l:
                    collapsed_seq = 0
                else:
                    collapsed_seq = 1
            elif 3 in l:
                if 4 in l:
                    collapsed_seq = 2
                else:
                    collapsed_seq = 3
            elif 6 in l:
                if 7 in l:
                    collapsed_seq = 4
                else:
                    collapsed_seq = 5
            elif 9 in l:
                if 4 in l:
                    collapsed_seq = 6
                else:
                    collapsed_seq = 7
            elif 10 in l:
                if 12 in l:
                    collapsed_seq = 8
                else:
                    collapsed_seq = 9
            elif 13 in l:
                if 14 in l:
                    collapsed_seq = 10
                else:
                    collapsed_seq = 11
            elif 16 in l:
                if 17 in l:
                    collapsed_seq = 12
                else:
                    collapsed_seq = 13
            else:
                if 14 in l:
                    collapsed_seq = 14
                else:
                    collapsed_seq = 15
            collapsed.append(collapsed_seq)
        return Tensor(collapsed).type(dtype=torch.long)

def converter(wav, sr, nmels=40, tweak=False, verbose=False):
    win_len = int(sr*0.025)
    hop = int(0.005*sr)
    tweaker = choice(['mask_time', 'mask_frequency', 'both_masks', 'noise'])
    mel_net = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=nmels, win_length=win_len, n_fft=win_len,
                                                hop_length=hop)
    if tweak:
        if tweaker != 'noise':
            tweak_net = []
            if tweaker == 'mask_time':
                tweak_net.append(torchaudio.transforms.TimeMasking(time_mask_param=20))
            elif tweaker == 'both_masks':
                tweak_net.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=10))
                tweak_net.append(torchaudio.transforms.TimeMasking(time_mask_param=15))
            else:
                tweak_net.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=10))
            tweak_net = nn.Sequential(*tweak_net)
        else:
            wav = wav + rand(wav.shape)*0.02
        if verbose:
            print(f'tweak: {tweaker}')
    converted = from_numpy(normalize(power_to_db(mel_net(wav)[0])))
    if tweak and tweaker != 'noise':
        converted = tweak_net(Tensor(converted))
    return converted

def converter_mfcc(wav, sr, nmfccs=15, tweak=False):
    win_len = 0.025
    hop = 0.005
    nfft = ceil(sr*win_len)
    if tweak:
        wav = wav + rand(wav.shape)*0.02
        wav = wav.numpy()
    else:
        wav = wav.numpy()
    mfccs = mfcc(wav, samplerate=sr, winlen=win_len, winstep=hop, numcep=nmfccs, nfft=nfft, appendEnergy=False,
                winfunc=np.hamming)
    converted = from_numpy(mfccs).transpose(0, 1)
    # normalise
    converted = converted - mean(converted, dim=1)[:, None]
    return converted.float()


def checkMel():
    filename = "pilot_2/TB/slow_sound_files/Lee_stale_12.wav"
    waveform, sampling_rate = torchaudio.load(filename)
    original = converter(waveform, sr=sampling_rate)
    tweaked = converter(waveform, sr=sampling_rate, tweak=True, verbose=True)
    fig, ax = plt.subplots(2)
    ax[0].imshow(original, aspect='auto', origin='lower')
    ax[1].imshow(tweaked, aspect='auto', origin='lower')


def checkMFCC():
    filename = "pilot_0/slow_sound_files/least_eel_7.wav"
    waveform, sampling_rate = torchaudio.load(filename)
    mfcc_original = converter_mfcc(waveform, sampling_rate, 15)
    mfcc_tweaked = converter_mfcc(waveform, sampling_rate, 15, tweak=True)
    fig, ax = plt.subplots(2)
    ax[0].imshow(mfcc_original, aspect='auto', origin='lower')
    ax[1].imshow(mfcc_tweaked, aspect='auto', origin='lower')


def loadData(file_path, val_ratio=0.15, train_tweak_ratio=0.3):
    # get paths to sounds
    slow_path = file_path + '/' + 'slow_sound_files'
    sped_path = file_path + '/' + 'sped_up'
    files = [slow_path + '/' + sound for sound in listdir(slow_path) if 'wav' in sound] + \
            [sped_path + '/' + sound for sound in listdir(sped_path) if 'wav' in sound]
    # shuffle data
    shuffle(files)
    # train validation split
    num_val = len(files)*val_ratio
    num_tweak = (len(files)-num_val)*train_tweak_ratio
    val = []
    train = []
    val_count = 0
    tweak_count = 0
    for sound in files:
        # get target sequence words
        target_words = sound.split('/')[-1]
        target_words = target_words.split('_')[:2]
        # get audio and sampling rate info
        wave, sampling_rate = torchaudio.load(sound)
        # for the validation split
        if val_count <= num_val:
            val.append((wave, sampling_rate, target_words))
            val_count += 1
        # for the train split
        else:
            # for the tweaked split, duplicate audio data and add augment indication tag
            if tweak_count <= num_tweak:
                train.append((wave, sampling_rate, target_words, 'tweak'))
                train.append((wave, sampling_rate, target_words, 'no_tweak'))
                tweak_count += 1
            # for the non tweaked split
            else:
                train.append((wave, sampling_rate, target_words, 'no_tweak'))
    return train, val


def loadNormal(normal_file_path):
    files = [normal_file_path + '/' + sound for sound in listdir(normal_file_path) if 'wav' in sound]
    shuffle(files)
    data = []
    for sound in files:
        target_words = sound.split('/')[-1]
        word_rep = (target_words.split('_')[-1]).strip('.wav')
        target_words = target_words.split('_')[:2]
        wave, sampling_rate = torchaudio.load(sound)
        data.append((wave, sampling_rate, target_words, word_rep))
    return data

def processNormal(data, data_type='mel_spec'):
    # initialise label converter
    labeller = LabelConvert()
    # initialise empty lists for data
    mel_specs = []
    labels = []
    reps = []
    if data_type == 'mel_spec':
        convert_func = converter
    else:
        convert_func = converter_mfcc
    for item in data:
        wave, sampling_rate, words, rep = item
        spec = convert_func(wave, sr=sampling_rate).transpose(0, 1)
        # append data
        mel_specs.append(spec)
        target = Tensor(labeller.words_to_labels(words))
        labels.append(target)
        reps.append(int(rep))
    # pad sequences in the batch
    mel_specs = nn.utils.rnn.pad_sequence(mel_specs, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True).int()
    reps = Tensor(reps).int()
    return mel_specs, labels, reps


def dataProcess(data, train=True, data_type='mel_spec'):
    # initialise label converter
    labeller = LabelConvert()
    # initialise empty lists for data
    mel_specs = []
    labels = []
    input_lens = []
    label_lens = []
    if data_type == 'mel_spec':
        convert_func = converter
    else:
        convert_func = converter_mfcc
    for item in data:
        # if process training set
        if train:
            wave, sampling_rate, words, aug = item
            if aug == 'tweak':
                spec = convert_func(wave, sr=sampling_rate, tweak=True).transpose(0, 1)
            else:
                spec = convert_func(wave, sr=sampling_rate).transpose(0, 1)
        # if process validation set
        else:
            wave, sampling_rate, words = item
            spec = convert_func(wave, sr=sampling_rate).transpose(0, 1)
        # append data
        mel_specs.append(spec)
        target = Tensor(labeller.words_to_labels(words))
        labels.append(target)
        label_lens.append(len(target))
        input_lens.append(spec.shape[0])
    # pad sequences in the batch
    mel_specs = nn.utils.rnn.pad_sequence(mel_specs, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True).int()
    input_lens = Tensor(input_lens).int()
    label_lens = Tensor(label_lens).int()
    return mel_specs, labels, input_lens, label_lens

def checkDataProcess():
    train_data, val_data = loadData('pilot_1')
    batch_size = 50
    train_loader = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                        collate_fn = lambda x: dataProcess(x))
    val_loader = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True,
                                    collate_fn = lambda x: dataProcess(x, train=False))
    for batch_num, data in enumerate(train_loader):
        spec, targets, input_lens, target_lens = data
        print(f'batch: {batch_num}\nspec shape: {spec.shape}\ntarget shape: {targets.shape}\n'
            f'input len shape: {input_lens.shape}\ntarget len shape: {target_lens.shape}')
    for batch_num, data in enumerate(val_loader):
        spec, targets, input_lens, target_lens = data
        print(f'batch: {batch_num}\nspec shape: {spec.shape}\ntarget shape: {targets.shape}\n'
            f'input len shape: {input_lens.shape}\ntarget len shape: {target_lens.shape}')

if __name__ == '__main__':
    checkMel()
    checkMFCC()
    checkDataProcess()
