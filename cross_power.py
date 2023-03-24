import numpy as np
from scipy import signal


# Создание двух звуковых сигналов
fs = 1000  # частота дискретизации
t = np.arange(0, 1, 1/fs)  # временная шкала
x1 = np.sin(2*np.pi*10*t)  # первый сигнал
x2 = np.sin(2*np.pi*20*t)  # второй сигнал

# Создание массива микрофонов
n_mics = 5  # количество микрофонов
microphones = np.random.randn(n_mics, len(x1))

# Вычисление кросс-спектра между парами сигналов микрофонов
nperseg = 256
n_mics_pairs = n_mics * (n_mics - 1) // 2
cross_spectra = np.zeros((n_mics_pairs, nperseg // 2 + 1), dtype=complex)
pair_index = 0
for i in range(n_mics):
    for j in range(i + 1, n_mics):
        f, cross_spectra[pair_index] = signal.csd(microphones[i], microphones[j], fs=fs, nperseg=nperseg)
        pair_index += 1

# Вычисление матрицы кросс-корреляции
cross_correlation_matrix = np.mean(np.abs(cross_spectra)**2, axis=-1).reshape(n_mics, n_mics)
for i in range(n_mics):
    for j in range(i + 1, n_mics):
        cross_correlation_matrix[j, i] = cross_correlation_matrix[i, j]

print("Cross-correlation matrix:")
