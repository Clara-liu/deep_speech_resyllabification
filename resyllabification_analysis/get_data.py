import pandas as pd
import scipy.io.wavfile as wav
from shutil import rmtree
from os import listdir, makedirs, path
from python_speech_features.base import logfbank

# read stimuli list
from resyllabification_study.stimuli import stimuli_pilot
stimuli_pilot = [x.replace(' ', '_') for x in stimuli_pilot]

class Stimuli:
    def __init__(self):
        # create dict for mapping pair number
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
        # create dict for mapping onset type (coda or onset)
        self.onset_dict = {}
        onset_end = ['e', 'o', 'r']
        for word in stimuli_pilot:
            first_word = word.split('_')[0]
            if first_word[-1] in onset_end:
                self.onset_dict[word] = 'Onset'
            else:
                self.onset_dict[word] = 'Coda'
        # create dict for mapping vowel targets (i or a)
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

def read_wav_files(folder_path: 'path to folder containing the wav files')-> 'dict with word as key and wav object as item':
    sound_files = [f'{folder_path}/{x}' for x in listdir(folder_path) if 'wav' in x]
    wav_dict = {file.split('.')[-2].split('/')[-1]: wav.read(file) for file in sound_files}
    return wav_dict

def subset_pair_condition(folder_path: 'path to folder to save the subsetted data',
                          all_speaker_df_path: 'path to all speaker data file'):
    # read combined df from all speakers
    combined_df = pd.read_csv(all_speaker_df_path, sep='\t')
    # get the unique pair numbers
    pairs = pd.unique(combined_df['Pair'])
    # get the unique conditions
    conditions = pd.unique(combined_df['Condition'])
    # create new folder for subsetted dfs
    new_folder_path = f'{folder_path}/byPair'
    if not path.exists(new_folder_path):
        makedirs(new_folder_path)
    else:
        rmtree(new_folder_path)
        makedirs(new_folder_path)
    # loop through the paris and conditions and subset combined df
    for p in pairs:
        for c in conditions:
            current_file_path = f'{new_folder_path}/P{p}_{c}.txt'
            current_df = combined_df[(combined_df['Pair'] == p) & (combined_df['Condition'] == c)]
            current_df.to_csv(current_file_path, sep='\t', index=False)


def get_mel_spec(token_folder_path: 'str path to folder containing all sound files for a speaker',
                 speaker: 'str which speakers',
                 nmel: 'int number of mel filter banks')-> 'pandas df for this speaker and this folder':
    stimuli_mapper = Stimuli()
    wav_file_dict = read_wav_files(token_folder_path)
    columns = ['c'+ str(x) for x in range(nmel)]
    for word, wav_data in wav_file_dict.items():
        #current_rep = str.zfill(word.split('_')[-1], 2)
        current_rep = int(word.split('_')[-1])
        current_words = '_'.join(word.split('_')[:2])
        sr = wav_data[0]
        sig = wav_data[1]
        mel_spec = logfbank(sig, sr, winlen=0.025, winstep=0.005, nfilt=nmel, nfft=int(sr*0.025))
        current_df = pd.DataFrame(data=mel_spec, columns=columns)
        current_df['Speaker'] = speaker
        current_df['Words'] = current_words
        current_df['Rep'] = current_rep
        current_df['Vowel'] = stimuli_mapper.get_vowel(current_words)
        current_df['Condition'] = stimuli_mapper.get_condition(current_words)
        current_df['Pair'] = stimuli_mapper.get_pair(current_words)
        if 'df' not in locals():
            df = current_df
        else:
            df = pd.concat([df, current_df])
    df.reset_index(inplace=True)
    return df

def get_all_speakers(speakers: 'list speakers', combined_df_path: 'str path to save combined df',
                     segmented_parent_folder: 'str resyllabified_condition or slow_sound_files',
                     nmel)-> 'save all speakers df':
    for s in speakers:
        file_path = f'../pilot_2/{s}/{segmented_parent_folder}/consonants'
        current_df = get_mel_spec(file_path, s, nmel)
        if 'all_speakers_df' not in locals():
            all_speakers_df = current_df
        else:
            all_speakers_df = pd.concat([all_speakers_df, current_df])
    all_speakers_df.to_csv(combined_df_path, sep='\t', index=False)

def main(speakers: 'str which speakers',
         path_to_combined: 'path to save the combined df',
         segmented_parent_folder: 'str resyllabified_condition or slow_sound_files',
         nmel: 'int number of mel spec bank',
         path_to_subsetted: 'path to subsetted data')-> 'save subsetted mel spec data by pair':
        get_all_speakers(speakers, path_to_combined, segmented_parent_folder, nmel)
        subset_pair_condition(path_to_subsetted, path_to_combined)

if __name__ == '__main__':
    main(['FE', 'BS', 'RB','MAG', 'SG', 'AR'],
         '../pilot_2/mel_data/all_speakers_resyllabified_consonants.txt',
         'resyllabified_condition',
         26,
         '../pilot_2/mel_data/resyllabified_conosnants')