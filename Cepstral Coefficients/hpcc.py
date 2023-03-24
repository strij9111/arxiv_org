"""
HPCC (High-Pass Cepstral Coefficients)
"""
import librosa
import numpy as np


def extract_hpcc(y, sr, n_fft=2048, hop_length=512, n_hpcc=20):
    # Вычисление MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_hpcc)

    # Вычисление первой производной MFCC (delta)
    delta_mfcc = librosa.feature.delta(mfcc)

    # Получение "high-pass" версии MFCC, вычитая первую производную
    hpcc = mfcc - delta_mfcc

    return hpcc


# Загрузка аудиофайла
audio_file = 'your_audio_file.wav'
y, sr = librosa.load(audio_file)

# Извлечение HPCC признаков
hpcc_features = extract_hpcc(y, sr)

print("HPCC features shape:", hpcc_features.shape)
