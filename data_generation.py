from torch import nn, from_numpy, Tensor, utils, rand
import torchaudio
from random import choice, shuffle
from librosa import power_to_db
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
torchaudio.set_audio_backend('sox_io')

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


def converter(wav, sr, nmels=128, tweak=False, verbose=False):
    win_len = int(sr*0.025)
    hop = int(0.005*sr)
    tweaker = choice(['mask_time', 'mask_frequency', 'both_masks', 'noise'])
    layers = [torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=nmels, win_length=win_len, n_fft=win_len,
                                                   hop_length=hop)]
    if tweak:
        if tweaker != 'noise':
            if tweaker == 'mask_time':
                layers.append(torchaudio.transforms.TimeMasking(time_mask_param=20))
            elif tweaker == 'both_masks':
                layers.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=15))
                layers.append(torchaudio.transforms.TimeMasking(time_mask_param=15))
            else:
                layers.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=20))
        else:
            wav = wav + rand(wav.shape)*0.02
        if verbose:
            print(f'tweak: {tweaker}')
    net = nn.Sequential(*layers)
    converted = from_numpy(power_to_db(net(wav)[0]))
    return converted


def checkMel():
    filename = "pilot/slow_sound_files/Lee_steal_5.wav"
    waveform, sampling_rate = torchaudio.load(filename)
    original = converter(waveform, sr=sampling_rate)
    tweaked = converter(waveform, sr=sampling_rate, tweak=True, verbose=True)
    fig, ax = plt.subplots(2)
    ax[0].imshow(original, aspect='auto', origin = 'lower')
    ax[1].imshow(tweaked, aspect='auto', origin = 'lower')


def loadData(file_path, val_ratio=0.15):
    # get paths to sounds
    slow_path = file_path + '/' + 'slow_sound_files'
    sped_path = file_path + '/' + 'sped_up'
    files = [slow_path + '/' + sound for sound in listdir(slow_path) if 'wav' in sound] + \
            [sped_path + '/' + sound for sound in listdir(sped_path) if 'wav' in sound]
    # shuffle data
    shuffle(files)
    # train validation split
    num_val = len(files)*val_ratio
    val = []
    train = []
    count = 1
    for sound in files:
        # get target sequence words
        target_words = sound.split('/')[-1]
        target_words = target_words.split('_')[:2]
        wave, sampling_rate = torchaudio.load(sound)
        data = (wave, sampling_rate, target_words)
        if count <= num_val:
            val.append(data)
        else:
            train.append(data)
        count += 1
    return train, val


def dataProcess(data, batch_size, train=True, train_tweak_ratio = 0.3):
    # initialise label converter
    labeller = LabelConvert()
    # initialise empty lists for data
    mel_specs = []
    labels = []
    input_lens = []
    label_lens = []
    if train:
        # get number of sample to augment according to defined ratio
        train_tweak_num = int(batch_size*train_tweak_ratio)
        # count for augmentation
        count = 1
    for (wave, sampling_rate, words) in data:
        # get original data
        spec = converter(wave, sr=sampling_rate).transpose(0, 1)
        mel_specs.append(spec)
        target = Tensor(labeller.words_to_labels(words))
        labels.append(target)
        label_lens.append(len(target))
        input_lens.append(spec.shape[0])
        if train and count <= train_tweak_num:
            # augmented data
            spec_tweaked = converter(wave, sr=sampling_rate, tweak=True).transpose(0, 1)
            mel_specs.append(spec_tweaked)
            labels.append(target)
            label_lens.append(len(target))
            input_lens.append(spec_tweaked.shape[0])
            count += 1
    # pad sequences in the batch
    mel_specs = nn.utils.rnn.pad_sequence(mel_specs, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lens = Tensor(input_lens)
    label_lens = Tensor(label_lens)
    return mel_specs, labels, input_lens, label_lens

def checkDataProcess():
    train_data, val_data = loadData('pilot')
    batch_size = 50
    train_loader = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                         collate_fn = lambda x: dataProcess(x, batch_size=batch_size))
    val_loader = utils.data.DataLoader(dataset=val_data, batch_size=50, shuffle=True,
                                       collate_fn = lambda x: dataProcess(x, batch_size=batch_size, train=False))
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
    checkDataProcess()