"""
извлечение voice perturbation features, таких как jitter, shimmer и HNR (Harmonic-to-Noise Ratio)
"""
import numpy as np
from pydub import AudioSegment
from scipy.signal import find_peaks

def calculate_jitter(pitch_periods):
    """Calculate jitter using the pitch periods."""
    jitter = np.abs(np.diff(pitch_periods)).sum() / len(pitch_periods)
    return jitter

def calculate_shimmer(amplitude_values):
    """Calculate shimmer using the amplitude values."""
    shimmer = np.abs(np.diff(amplitude_values)).sum() / len(amplitude_values)
    return shimmer

def calculate_hnr(signal, pitch_periods):
    """Calculate HNR using the signal and pitch periods."""
    harmonic_signal = np.zeros_like(signal)
    for peak in pitch_periods:
        harmonic_signal += signal[peak::peak]
    noise_signal = signal - harmonic_signal
    hnr = 20 * np.log10(np.abs(harmonic_signal).sum() / np.abs(noise_signal).sum())
    return hnr

def extract_voice_perturbation_features(audio_file):
    """Extract jitter, shimmer, and HNR from an audio file."""
    audio = AudioSegment.from_file(audio_file)
    samples = np.array(audio.get_array_of_samples())
    pitch_periods = find_peaks(samples)[0]
    amplitude_values = samples[pitch_periods]

    jitter = calculate_jitter(pitch_periods)
    shimmer = calculate_shimmer(amplitude_values)
    hnr = calculate_hnr(samples, pitch_periods)

    return jitter, shimmer, hnr
