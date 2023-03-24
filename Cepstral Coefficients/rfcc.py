"""
RFCC (Relative Frequency Cepstral Coefficients)
"""
import numpy as np
import librosa
import scipy.fftpack as fft

def rfcc(y, sr, n_rfcc=20, n_fft=2048, hop_length=512, win_length=None):
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    power_spectrogram = D**2
    relative_spectrum = power_spectrogram / np.sum(power_spectrogram, axis=0)
    log_relative_spectrum = np.log10(relative_spectrum + 1e-10)
    rfcc_feat = fft.dct(log_relative_spectrum, type=2, axis=0, norm='ortho')[:n_rfcc]
    return rfcc_feat
