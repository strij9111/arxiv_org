import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import scipy.io.wavfile as wavfile


def spectral_subtraction(noisy_signal, noise_estimate, alpha=2.0):
    magnitude_noisy = np.abs(noisy_signal)
    phase_noisy = np.angle(noisy_signal)

    magnitude_clean = magnitude_noisy - alpha * noise_estimate
    magnitude_clean = np.maximum(magnitude_clean, 0)

    clean_signal = magnitude_clean * np.exp(1j * phase_noisy)
    return clean_signal


def variable_step_lms(noisy_signal, desired_signal, initial_step_size=0.01, mu=0.99):
    num_samples = len(noisy_signal)
    weights = np.zeros(num_samples, dtype=np.complex128)
    output = np.zeros(num_samples, dtype=np.complex128)
    step_size = initial_step_size

    for i in range(num_samples):
        output[i] = np.dot(weights, noisy_signal[i:])
        error = desired_signal[i] - output[i]
        step_size = mu * step_size + (1 - mu) * np.abs(error) ** 2
        weights += step_size * np.conj(error) * noisy_signal[i:]

    return output

def speech_enhancement(noisy_signal, noise_estimate, frame_length=256, frame_overlap=0.5, alpha=2.0, initial_step_size=0.01, mu=0.99):
    noisy_signal = np.array(noisy_signal)
    num_frames = int(len(noisy_signal) / (frame_length * (1 - frame_overlap)))

    window = signal.windows.hann(frame_length)
    enhanced_signal = np.zeros_like(noisy_signal, dtype=np.complex128)

    for frame_idx in range(num_frames):
        frame_start = int(frame_idx * frame_length * (1 - frame_overlap))
        frame_end = frame_start + frame_length

        noisy_frame = noisy_signal[frame_start:frame_end] * window
        noise_frame = noise_estimate[frame_start:frame_end] * window

        noisy_frame_fft = fftpack.fft(noisy_frame)
        noise_frame_fft = fftpack.fft(noise_frame)

        enhanced_frame_fft = spectral_subtraction(noisy_frame_fft, noise_frame_fft, alpha)
        enhanced_frame = fftpack.ifft(enhanced_frame_fft)

        enhanced_signal[frame_start:frame_end] += np.real(enhanced_frame)

    desired_signal = enhanced_signal.copy()
    enhanced_signal = variable_step_lms(noisy_signal, desired_signal, initial_step_size, mu)

    return np.real(enhanced_signal)


# Загрузите зашумленный аудиосигнал и шумовую оценку (это должны быть одномерные массивы)
fs, noisy_signal = wavfile.read("noisy_speech.wav")
fs, noise_estimate = wavfile.read("noise_estimate.wav")

# Убедитесь, что сигналы одинаковой длины
if len(noisy_signal) > len(noise_estimate):
    noisy_signal = noisy_signal[:len(noise_estimate)]
else:
    noise_estimate = noise_estimate[:len(noisy_signal)]

# Выполните алгоритм улучшения речи
enhanced_signal = speech_enhancement(noisy_signal, noise_estimate)
# Нормализуйте усиленный сигнал и преобразуйте его в int16
enhanced_signal = (enhanced_signal / np.max(np.abs(enhanced_signal)) * np.iinfo(np.int16).max).astype(np.int16)

# Сохраните усиленный сигнал в файл формата WAV
wavfile.write("enhanced_speech.wav", fs, enhanced_signal)
