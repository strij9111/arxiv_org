"""
Perceptual Linear Prediction
"""
import numpy as np
from pydub import AudioSegment
from pydub.utils import audioop
import scipy.fftpack as fft
import scipy.signal as signal


def compute_plp(audio_file):
    audio = AudioSegment.from_file(audio_file)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    # Параметры PLP
    frame_size = 0.025
    frame_shift = 0.010
    n_filters = 13

    # Вычисление PLP
    frames = librosa.util.frame(samples, int(frame_size * sample_rate), int(frame_shift * sample_rate)).T
    plp = np.empty((0, n_filters))

    for frame in frames:
        plp_frame = audioop.lin2alaw(frame.tobytes(), audio.sample_width)
        plp_frame = audioop.alaw2lin(plp_frame, audio.sample_width)

        # Применение окна Хэмминга и преобразование Фурье
        windowed = signal.windows.hamming(len(plp_frame)) * plp_frame
        plp_frame_fft = np.abs(fft.fft(windowed))[:len(plp_frame) // 2]

        # Вычисление PLP-коэффициентов
        plp_coeffs = librosa.feature.mfcc(S=librosa.power_to_db(plp_frame_fft), n_mfcc=n_filters)
        plp = np.vstack((plp, plp_coeffs))

    return plp


audio_file = 'path/to/your/audiofile.wav'
plp_feat = compute_plp(audio_file)
