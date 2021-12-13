from dtw_analysis import sound_files
from chopping_analysis.get_mfcc import Stimuli
import pandas as pd

def get_normal_duration(speakers: 'list'):
    # initialise empty dict for data
    data = {'Speaker':[],
            'Pair': [],
            'Rep': [],
            'Condition': [],
            'Resyllabification': [],
            'Duration': [],
            'Vowel': []}
    # initialise Stimuli instance
    converter = Stimuli()
    # loop through speakers
    for s in speakers:
        file_paths = [f'pilot_2/{s}/normal_sound_files/non_resyllabified',
                      f'pilot_2/{s}/normal_sound_files/resyllabified',
                      f'pilot_2/{s}/normal_sound_files']
        # loop through directories
        for path in file_paths:
            # initialise sound_file instance
            sound_obj = sound_files(s, {0:path})
            # read all the wav files in the current directory
            sound_obj.get_wavs()
            # loop through all the wav files in the current directory
            for word, wav in sound_obj.wav_dict.items():
                words = '_'.join(word.split('_')[:2])
                # onset or coda
                condition = converter.get_condition(words)
                # only get the duration data if it is the onset condition or not miss classified at the incorrect word
                if len(path.split('/')) == 4 or condition == 'Onset':
                    rep = word.split('_')[-2]
                    pair = converter.get_pair(words)
                    resyllab = 1 if path.split('/')[-1] == 'resyllabified' else 0
                    duration = wav.shape[0]/sound_obj.sr
                    vowel = converter.get_vowel(words)
                    # append data
                    data['Speaker'].append(s)
                    data['Pair'].append(pair)
                    data['Rep'].append(rep)
                    data['Condition'].append(condition)
                    data['Resyllabification'].append(resyllab)
                    data['Duration'].append(duration)
                    data['Vowel'].append(vowel)
    return pd.DataFrame.from_dict(data)