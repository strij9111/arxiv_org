"""
LPCC (Linear Prediction Cepstral Coefficients)
"""
import numpy as np
import librosa
from scipy.signal import lfilter

def lpcc(y, sr, n_lpcc=20, n_fft=2048, hop_length=512, win_length=None):
    y_frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)
    autocorr = np.apply_along_axis(librosa.core.autocorrelate, 0, y_frames)
    lpc_coeffs = np.apply_along_axis(librosa.core.lpc, 0, autocorr, n_lpcc)
    lpcc_feat = lfilter([0] + -1 * lpc_coeffs[1:], [1], lpc_coeffs)
    return lpcc_feat
