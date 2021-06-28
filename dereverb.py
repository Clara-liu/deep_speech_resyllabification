import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft
import os

sounds_dir = 'pilot_2/FE/normal_sound_files'


def dereverb(dir, nchannel, sr, delay=3, iteration=6, taps=10):
    new_dir = f'{dir}/dereverbed'
    os.mkdir(new_dir)
    files = [dir + '/' + sound for sound in os.listdir(dir) if 'wav' in sound]
    for f in files:
        signal_list = [sf.read(f)[0] for d in range(nchannel)]
        y = np.stack(signal_list, axis=0)
        stft_options = dict(size=int(sr*0.025), shift=int(sr*0.005))
        Y = stft(y, **stft_options).transpose(2, 0, 1)
        Z = wpe(Y, taps=taps, delay=delay, iterations=iteration, statistics_mode='full').transpose(1, 2, 0)
        z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])
        write(f'{new_dir}/{f.split("/")[-1]}', sr, z[0])


if __name__ == '__main__':
    dereverb(sounds_dir, 1, 32000)