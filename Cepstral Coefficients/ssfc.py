"""
Sub-band Spectral Flux Coefficients
"""

import numpy as np
import librosa

def ssfc(y, sr, n_bands=20, n_fft=2048, hop_length=512, win_length=None):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_bands)
    mel_spectrum = np.dot(mel_basis, S)
    spectral_flux = librosa.onset.onset_strength(S=mel_spectrum, sr=sr, hop_length=hop_length)
    ssfc_feat = spectral_flux
    return ssfc_feat
