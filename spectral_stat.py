import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt

# Загрузка аудиофайла
audio_file = 'your_audio_file.wav'
y, sr = librosa.load(audio_file)

# Вычисление преобразования Фурье (FFT) для аудиосигнала
n_fft = 2048
spectrogram = np.abs(librosa.stft(y, n_fft=n_fft))

# Вычисление мощностного спектра
power_spectrogram = spectrogram**2

# Получение формы спектра
spectral_shape = np.mean(power_spectrogram, axis=1)

# Вычисление наклона спектра
freqs = np.linspace(0, sr / 2, num=len(spectral_shape))
slope, intercept = np.polyfit(freqs, 10 * np.log10(spectral_shape), 1)

# Построение спектра и его наклона
plt.figure()
plt.plot(freqs, 10 * np.log10(spectral_shape), label='Spectral shape')
plt.plot(freqs, slope * freqs + intercept, label='Slope', linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.legend()
plt.show()

# Статистические характеристики спектра
mean = np.mean(spectral_shape)
std_dev = np.std(spectral_shape)
min_val = np.min(spectral_shape)
max_val = np.max(spectral_shape)

print("Mean:", mean)
print("Standard deviation:", std_dev)
print("Minimum value:", min_val)
print("Maximum value:", max_val)
print("Spectral slope:", slope)
