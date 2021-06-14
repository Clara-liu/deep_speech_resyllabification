from python_speech_features import mfcc
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os.path
import numpy as np


class wavFiles:
    group_dict = {'0': ('Lee_steal', 'Lee_stale', 'least_eel', 'least_ale'),
                    '1': ('Kerr_speel', 'Kerr_spale', 'cusp_eel', 'cusp_ale'),
                    '2': ('do_mart', 'do_meet', 'doom_art', 'doom_eat'),
                    '3': ('coo_part', 'coo_Pete', 'coop_art', 'coop_eat')}

    def __init__(self, group_num):
        self.words = self.group_dict[group_num]
        self.wav_dict = {}
        self.sr = 0

    def get_files(self, path):
        for word in self.words:
            self.wav_dict[word] = []
            for i in np.arange(1, 11):
                rep = str(i)
                file_path = path + '/' + word + '_' + rep + '.wav'
                if os.path.isfile(file_path):
                    rate, file = wav.read(file_path)
                    if i == 1 and word == self.words[0]:
                        self.sr = rate
                    self.wav_dict[word].append(file)

def getMFCC(wav: 'wavFiles object', speaker: 'str which speaker', nmfcc: ' int number of mfcc',
            group: 'str group code')->'pandas mfcc dataframe':
    # get sampling rate of sig
    sr = wav.sr
    words = wav.words
    # count number of FFT
    n_fft = int(sr*0.025)
    # create col names
    columns = ['c'+ str(x) for x in range(nmfcc)]
    # get mfcc data for each sequence
    for word in words:
        for i in range(len(wav.wav_dict[word])):
            sig = wav.wav_dict[word][i]
            rep = str.zfill(str(i+1), 2)
            features = mfcc(sig, samplerate=sr, winlen=0.025, winstep=0.005, numcep=nmfcc, nfft=n_fft,
                            preemph=0.97, ceplifter=22, appendEnergy=False, winfunc=np.hamming)

            current_data = pd.DataFrame(data=features, columns=columns)
            current_data['Speaker'] = speaker
            current_data['Word'] = word
            current_data['Rep'] = rep
            current_data['Group'] = group
            if words.index(word) == 0 and rep == '01':
                data = current_data
            else:
                data = pd.concat([data, current_data])

    # create new col from old index and rest index
    data.reset_index(inplace=True)
    return data

def plotMFCC(data: 'pandas mfcc dataframe', word: 'str which word', rep: 'str which repetition',
             speaker: 'str which speaker')->None:
    df = data[(data.Word == word) & (data.Rep == rep)& (data.Speaker == speaker)].reset_index(drop=True)
    df = df.select_dtypes('number').to_numpy()
    plot_data = np.swapaxes(df, 0, 1)
    plt.figure(figsize=(15, 5))
    plt.title('MFCCs for {0}{1}'.format(word, rep))
    plt.imshow(plot_data, aspect='auto', origin = 'lower')

def prepData(df: 'pandas mfcc dataframe', sr: 'float sampling rate of mfcc df',
             train_rep: 'int number of repetition in the train set',
             get_difference : 'bool return derivative or not' = True) -> 'train/test data':
    # create dict for categorical encoding
    word_dict = {w: i for i, w in enumerate(df.Word.unique())}
    # scaler for difference data
    scaler = MinMaxScaler((-50, 50))
    # calculate first difference
    if get_difference:
        diff_df = df.select_dtypes('float').diff(periods=1, axis=0)
        diff_df[df['index'] == 0] = 0
        diff_df = diff_df/sr
        diff_df = scaler.fit_transform(diff_df)
        diff_df = pd.DataFrame(data=diff_df, columns=[str(x) + '_diff' for x in range(diff_df.shape[1])])
        df = pd.concat([df, diff_df], axis=1)
    # get smallest row count among all sequences
    seq_len = df.groupby(['Word', 'Rep', 'Speaker']).agg('count').c0.min()
    # trim each sequence
    df = df[df['index']<seq_len]
    # subset for train, val, test randomly
    reps = df.Rep.unique()
    np.random.shuffle(reps)
    nfeat = df.select_dtypes('float').shape[1]

    train = df[df['Rep'].isin(reps[:train_rep])]
    nseq_train = int(train.shape[0]/seq_len)
    p = np.random.permutation(nseq_train)
    x_train = train.select_dtypes('float').to_numpy()
    x_train = np.reshape(x_train, (nseq_train, seq_len, nfeat))
    x_train = x_train[p, :, :]
    y_train = np.array(list(map(word_dict.get, train.Word[train['index'] == 0])))
    y_train = np.reshape(y_train, (nseq_train, 1))
    y_train = y_train[p, :]

    test = df[df['Rep'].isin(reps[train_rep:])]
    nseq_test = int(test.shape[0]/seq_len)
    p = np.random.permutation(nseq_test)
    x_test = test.select_dtypes('float').to_numpy()
    x_test = np.reshape(x_test, (nseq_test, seq_len, nfeat))
    x_test = x_test[p, :, :]
    y_test = np.array(list(map(word_dict.get, test.Word[test['index'] == 0])))
    y_test = np.reshape(y_test, (nseq_test, 1))
    y_test = y_test[p, :]

    data = {'train': (x_train, y_train), 'test': (x_test, y_test),
            'word_code': word_dict}
    return data
