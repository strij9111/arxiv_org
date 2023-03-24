"""
Gammatone Frequency Cepstral Coefficients (GFCC)
"""
import numpy as np
import librosa
import scipy.signal


def gammatone(sr, fmin, fmax, n_filters):
    frequencies = np.logspace(np.log10(fmin), np.log10(fmax), num=n_filters, base=10.0)
    widths = np.zeros(n_filters)
    for i in range(len(widths)):
        widths[i] = 1.019 * 24.7 * (4.37 * frequencies[i] / 1000 + 1)
    return frequencies, widths

def gfcc(y, sr, n_filters=64, n_coefficients=13, fmin=50, fmax=None):
    if fmax is None:
        fmax = sr / 2

    frequencies, widths = gammatone(sr, fmin, fmax, n_filters)
    filters = librosa.filters.gammatone(sr, frequencies, widths)

    power_spectrogram = np.abs(librosa.stft(y))**2
    power_spectrogram_filtered = np.dot(filters, power_spectrogram)

    log_power_spectrogram_filtered = np.log10(power_spectrogram_filtered + np.finfo(float).eps)
    gfccs = scipy.fftpack.dct(log_power_spectrogram_filtered, axis=0, type=2, norm='ortho')[:n_coefficients]

    return gfccs

y, sr = librosa.load('your_audio_file.wav', sr=None)
gfccs = gfcc(y, sr)
