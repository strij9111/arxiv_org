"""
Sub-band Centroid Magnitude Coefficients
"""
import numpy as np
import librosa

def scmc(y, sr, n_bands=20, n_fft=2048, hop_length=512, win_length=None):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_bands)
    mel_spectrum = np.dot(mel_basis, S)
    centroids = librosa.feature.spectral_centroid(S=mel_spectrum, sr=sr, n_fft=n_fft, hop_length=hop_length, freq=None)
    scmc_feat = centroids
    return scmc_feat
