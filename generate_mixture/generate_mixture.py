import os
import librosa
import numpy as np
import pyloudnorm as pyln
import random 
import soundfile as sf
# from pysndfx import AudioEffectChain
MAX_LOUDNESS = -25.
MIN_LOUDNESS = -33.
def load_mixture_list(dir_path):
    files = []
    for file in os.listdir(dir_path):
        # file_path = dir_path + file
        # audio, _ = librosa.load(file_path)
        files.append(file)
    return files

def get_random_loud():
    return random.random() * 8 - 33.

def load_and_remix(files, _dir):
    subdirs = ['s1', 's2', 'mix']
    subpaths = [_dir + '/' + subdir + '/' for subdir in subdirs]
    # 加载数据
    for file in files:
        audio_path = [subpath + file for subpath in subpaths]
        audios = [sf.read(path)[0] for path in audio_path]
        l1, l2 = get_random_loud(), get_random_loud()
        mixture = reset_loudness(audios[0], l1) + reset_loudness(audios[1], l2)
        sf.write(audio_path[2], mixture, 16000)


def reset_loudness(audio, dest_loudness):
    loudness = pyln.Meter(16000).integrated_loudness(audio)
    audio_ = pyln.normalize.loudness(audio, loudness, dest_loudness)
    return audio_

if __name__ == '__main__':
    for subdir in ['cv', 'tt', 'tr']:
        _dir = '/home/photon/Datasets/av-datas/lrs2_rebuild/audio/wav16k/min/' + subdir
        files = load_mixture_list(_dir + '/mix')
        load_and_remix(files, _dir)
    # print(load_mixture_list('/home/photon/Datasets/av-datas/lrs2_rebuild/audio/wav16k/min/cv/mix'))