from get_mfcc import Stimuli, match_onset, read_wav_files, subset_pair_condition
from python_speech_features.base import logfbank
import pandas as pd

# read stimuli list
from resyllabification_study.stimuli import stimuli_pilot
stimuli_pilot = [x.replace(' ', '_') for x in stimuli_pilot]


def get_mel_spec(token_folder_path: 'str path to folder containing all sound files for a speaker',
                 speaker: 'str which speakers',
                 nmel: 'int number of mel filter banks')-> 'pandas df for this speaker and this folder':
    stimuli_mapper = Stimuli()
    wav_file_dict = read_wav_files(token_folder_path)
    columns = ['c'+ str(x) for x in range(nmel)]
    for word, wav_data in wav_file_dict.items():
        current_rep = str.zfill(word.split('_')[-1], 2)
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

def get_all_speakers_mel_slow(speakers: 'list speakers', combined_df_path: 'str path to save combined df',
                              nmel)-> 'save all speakers df':
    for s in speakers:
        current_df = get_mel_spec(f'../pilot_2/{s}/slow_sound_files', s, nmel)
        if 'all_speakers_df' not in locals():
            all_speakers_df = current_df
        else:
            all_speakers_df = pd.concat([all_speakers_df, current_df])
    all_speakers_df.to_csv(combined_df_path, sep='\t', index=False)


def main(speakers: 'str which speakers',
         path_to_combined: 'path to save the combined df',
         resyllabification_condition: 'str resyllabified or non_resyllabified',
         speech_rate_condition: 'slow or normal',
         nmel: 'int number of mel spec bank',
         path_to_subsetted: 'path to subsetted data')-> 'save subsetted mel spec data by pair':
    if speech_rate_condition == 'normal':
        # placeholder for normal speech rate
        pass
    else:
        get_all_speakers_mel_slow(speakers, path_to_combined, nmel)
        subset_pair_condition(path_to_subsetted, path_to_combined)

if __name__ == '__main__':
    main(['FE', 'BS', 'RB', 'GJ', 'TB', 'MAG', 'SG', 'AR'],
         '../pilot_2/mel_data/all_speakers_slow_rate.txt',
         'non_resyllabified',
         'slow',
         26,
         '../pilot_2/mel_data/slow_rate')