"""
извлечению акустических признаков с использованием двухуровневой системы Mel
и перекрывающегося блочного преобразования (OBT)
"""
import numpy as np
import librosa
import scipy.signal

def imfcc1(mfcc):
    n_mfcc = mfcc.shape[0]
    imfcc = np.zeros_like(mfcc)

    for i in range(n_mfcc):
        imfcc[i] = (-1) ** i * mfcc[i]

    return imfcc

def two_level_mfcc_obt(y, sr, n_mfcc=20, n_fft=2048, hop_length=512, n_blocks=4, overlap_ratio=0.5):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    imfcc = imfcc1(mfcc)

    block_size = int(n_mfcc / n_blocks)
    overlap_size = int(block_size * overlap_ratio)

    mfcc_obt = np.zeros((n_mfcc, mfcc.shape[1]))
    imfcc_obt = np.zeros((n_mfcc, imfcc.shape[1]))

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size

        if i > 0:
            mfcc_obt[start:end] = mfcc[start - overlap_size:end + overlap_size]
            imfcc_obt[start:end] = imfcc[start - overlap_size:end + overlap_size]
        else:
            mfcc_obt[start:end] = mfcc[start:end]
            imfcc_obt[start:end] = imfcc[start:end]

    return mfcc_obt, imfcc_obt

