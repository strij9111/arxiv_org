import numpy as np
import scipy.signal as signal

# Пример данных сигналов от двух микрофонов
signal1 = np.random.rand(1000)
signal2 = np.roll(signal1, 50)  # Имитация задержки во времени на 50 отсчетов

# Функция для расчета временной задержки между двумя сигналами
def time_delay_estimate(signal1, signal2):
    cross_correlation = signal.correlate(signal1, signal2)
    delay = np.argmax(cross_correlation) - (len(signal1) - 1)
    return delay

# Функция для локализации источника звука с использованием алгоритма SRP-PHAT
# d - расстояние между микрофонами, м
# c - скорость звука, м/c
def estimate_azimuth_angle(mic1_signal, mic2_signal, sample_rate, d, c=343):
    n_fft = 1024
    overlap = n_fft // 2

    mic1_spec, _, _ = signal.stft(mic1_signal, fs=sample_rate, nperseg=n_fft, noverlap=overlap)
    mic2_spec, _, _ = signal.stft(mic2_signal, fs=sample_rate, nperseg=n_fft, noverlap=overlap)

    cross_spectrum = mic1_spec * np.conj(mic2_spec)
    phase_difference = np.angle(cross_spectrum)

    frequency_bins = np.linspace(0, sample_rate, n_fft)
    tau = phase_difference / (2 * np.pi * frequency_bins)

    tau_mean = np.mean(tau, axis=0)
    tau_std = np.std(tau, axis=0)
    valid_bins = np.where(tau_std < 0.5 * tau_mean)

    tau = np.mean(tau[:, valid_bins], axis=1)

    azimuth_angle = np.arcsin(c * tau / d)
    return azimuth_angle

# Расчет временной задержки и угла
delay = time_delay_estimate(signal1, signal2)

print("Time delay:", delay, "seconds")

# Применение функции localize_source
angle = estimate_azimuth_angle(signal1, signal2, 16000, 1)
print("Localized azimuth angle:", angle, "degrees")

