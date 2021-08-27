import pandas as pd
import scipy.io.wavfile as wav
from sklearn.preprocessing import MinMaxScaler
import os.path
from python_speech_features import mfcc

from resyllabification_study.stimuli import stimuli_pilot
stimuli_pilot = [x.replace(' ', '_') for x in stimuli_pilot]


class Stimuli:
    def __init__(self):
        self.pair_dict = {}
        for i in range(len(stimuli_pilot)):
            if i <= 3:
                self.pair_dict[stimuli_pilot[i]] = 0
            elif i <= 7:
                self.pair_dict[stimuli_pilot[i]] = 1
            elif i <= 11:
                self.pair_dict[stimuli_pilot[i]] = 2
            else:
                self.pair_dict[stimuli_pilot[i]] = 3
        self.onset_dict = {}
        onset_end = ['e', 'o', 'r']
        for word in stimuli_pilot:
            first_word = word.split('_')[0]
            if first_word[-1] in onset_end:
                self.onset_dict[word] = 'Onset'
            else:
                self.onset_dict[word] = 'Coda'
        self.vowel_dict = {}
        for ii in range(len(stimuli_pilot)):
            if ii % 2 == 0:
                self.vowel_dict[stimuli_pilot[ii]] = 'i'
            else:
                self.vowel_dict[stimuli_pilot[ii]] = 'a'

    def get_pair(self, w):
        return self.pair_dict[w]

    def get_condition(self, w):
        return self.onset_dict[w]

    def get_vowel(self, w):
        return self.vowel_dict[w]


def match_onset(speaker, folder_path, matching_path):
    stimuli = Stimuli()
    stimuli_df = pd.DataFrame.from_dict({'Word': stimuli_pilot,
                                         'Pair': stimuli.pair_dict.values(),
                                         'Vowel': stimuli.vowel_dict.values(),
                                         'Condition': stimuli.onset_dict.values()})







