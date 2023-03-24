"""
IMFCC является вариацией MFCC, которая включает обратное мел-преобразование.
"""
import numpy as np
import librosa

def mel2hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def imfcc(y, sr, n_mfcc=20, n_mels=128, fmin=0, fmax=None):
    # Вычисление MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Обратное мел-преобразование
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    hz_freqs = mel2hz(mel_freqs)

    imfcc_matrix = np.zeros((n_mfcc, len(hz_freqs)))

    for i in range(n_mfcc):
        imfcc_matrix[i, :] = hz2mel(hz_freqs) * mfcc[i, :]

    return imfcc_matrix

audio_file = 'path/to/your/audiofile.wav'
y, sr = librosa.load(audio_file)
imfcc_feat = imfcc(y, sr)
