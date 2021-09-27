import pandas as pd
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
import numpy as np
from shutil import copyfile, rmtree
from os import listdir, makedirs, path
from random import choice
from python_speech_features import mfcc

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


def match_onset(folder_path: 'path to the folder that needs matching',
                matching_path: 'path to the new folder containing the matching tokens for chopping analysis',
                normal_sounds_path: 'path to the folder containing all the tokens for the normal condition'):
    # create folder for the matching onset tokens
    if not path.exists(matching_path):
        makedirs(matching_path)
    else:
        rmtree(matching_path)
        makedirs(matching_path)
    # get the list of what to match
    match_targets = listdir(folder_path)
    for target in match_targets:
        current_target = '_'.join(target.split('_')[:2])
        # the index of the matching onset condition is 2 before the current coda condition
        current_match_idx = stimuli_pilot.index(current_target) - 2
        # identify the current matching words
        current_match = stimuli_pilot[current_match_idx]
        # get a list of the matching candidate onset tokens that exist for this speaker
        matching_candidates = [x for x in listdir(normal_sounds_path) if current_match in x
                               and x not in listdir(matching_path)]
        if len(matching_candidates) != 0:
            chosen_match = choice(matching_candidates)
            current_match_path = f'{normal_sounds_path}/{chosen_match}'
            # copy the matching sound file to the new folder
            copyfile(current_match_path, f'{matching_path}/{chosen_match}')
            # copy the resyllabified sound file to the new folder
            copyfile(f'{folder_path}/{target}', f'{matching_path}/{target}')


def read_wav_files(folder_path: 'path to folder containing the wav files')-> 'dict with word as key and wav object as item':
    sound_files = [f'{folder_path}/{x}' for x in listdir(folder_path) if 'wav' in x]
    wav_dict = {file.split('.')[-2].split('/')[-1]: wav.read(file) for file in sound_files}
    return wav_dict


def get_mfcc(folder_path: 'folder containing the tokens for chopping analysis.i.e. ../pilot_2/BS/resyllabified_condition',
             speaker: 'str which speaker i.e. BS',
             nmfcc: 'int number of mfcc to extract',
             center: 'whether or not to center the mfcc data' = True)-> 'pandas df for this speaker and this folder':
    # create stimuli class object
    stimuli_mapper = Stimuli()
    # get the dict for all the files in the folder
    wav_file_dict = read_wav_files(folder_path)
    # create list for the mfcc column names
    columns = ['c'+ str(x) for x in range(nmfcc)]
    # loop through the files and create the mfcc dfs
    for word, wav_file in wav_file_dict.items():
        current_rep = str.zfill(word.split('_')[-1], 2)
        current_words = '_'.join(word.split('_')[:2])
        sr = wav_file[0]
        sig = wav_file[1]
        mfccs = mfcc(sig, samplerate=sr, winlen=0.025, winstep=0.005, numcep=nmfcc, nfft=int(sr*0.025), preemph=0.97,
                     ceplifter=22, appendEnergy=False, winfunc=np.hamming)
        current_df = pd.DataFrame(data=mfccs, columns=columns)
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
    # center mfcc
    if center:
        scaler = StandardScaler(with_std=False)
        df[columns] = scaler.fit_transform(df[columns])
    return df



# this function is for the normal speech rate condition
def get_all_speakers_mfcc(speakers: 'list speakers', combined_df_path: 'path to the combined data file',
                          condition: 'str resyllabified or non_resyllabified', nmfcc) -> 'save all speakers df':
    for s in speakers:
        match_onset(f'../pilot_2/{s}/normal_sound_files/{condition}', f'../pilot_2/{s}/{condition}_condition',
                    f'../pilot_2/{s}/normal_sound_files')
        current_df = get_mfcc(f'../pilot_2/{s}/{condition}_condition', s, nmfcc)
        if speakers.index(s) == 0:
            all_speakers_df = current_df
        else:
            all_speakers_df = pd.concat([all_speakers_df, current_df])
    all_speakers_df.to_csv(combined_df_path, sep='\t', index=False)




# this function is for the slow speech rate condition
def get_all_speakers_mfcc_slow(speakers: 'list speakers',
                               combined_df_path: 'path to the combined data file', nmfcc) -> 'save all speakers df':
    for s in speakers:
        current_df = get_mfcc(f'../pilot_2/{s}/slow_sound_files', s, nmfcc)
        if speakers.index(s) == 0:
            all_speakers_df = current_df
        else:
            all_speakers_df = pd.concat([all_speakers_df, current_df])
    all_speakers_df.to_csv(combined_df_path, sep='\t', index=False)





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


def main(speakers: 'list of speakers',
         path_to_combined: 'path to save the combined df',
         condition: 'resyllabified or non_resyllabified',
         speech_rate_condition: 'slow or normal',
         nmfcc: 'int number of mfcc to extract',
         path_to_subsetted):
    if speech_rate_condition == 'normal':
        get_all_speakers_mfcc(speakers, path_to_combined, condition, nmfcc)
        subset_pair_condition(path_to_subsetted, path_to_combined)
    else:
        get_all_speakers_mfcc_slow(speakers, path_to_combined, nmfcc)
        subset_pair_condition(path_to_subsetted, path_to_combined)

if __name__ == '__main__':
    main(['FE', 'BS', 'RB', 'GJ', 'TB'],
         '../pilot_2/mfcc_data/all_speakers_slow_rate.txt',
         'non_resyllabified',
         'slow',
         15,
         '../pilot_2/mfcc_data/slow_rate')



