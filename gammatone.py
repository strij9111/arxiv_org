"""
В этом примере мы загружаем аудиофайл с помощью librosa.load(), вычисляем STFT, извлекаем фазовую информацию,
применяем гамматонный фильтрбанк, вычисляем первую и вторую производные фазового спектрограммы по времени
и частоте, а также рассчитываем непрерывность фазы.
"""
import librosa
import numpy as np

# Загрузка аудиофайла
audio_file = "path/to/your/audiofile.wav"
y, sr = librosa.load(audio_file, sr=None)

# Вычисление коротковременного преобразования Фурье (STFT)
stft = librosa.stft(y)

# Извлечение фазовой информации из STFT
phase = np.angle(stft)

# Гамматонная фильтровальная банка
n_filters = 20
fmin = 50
fmax = 2000
filter_bank = librosa.filters.mel(sr, n_fft=2048, n_mels=n_filters, fmin=fmin, fmax=fmax, htk=True)

# Применение гамматонной фильтровальной банки к фазовому спектрограмме
gammatone_phase_spectrogram = filter_bank.dot(phase)

# Вычисление первой и второй производных по времени и частоте
dphase_dt = np.diff(gammatone_phase_spectrogram, axis=1)
dphase_df = np.diff(gammatone_phase_spectrogram, axis=0)
ddphase_dt2 = np.diff(dphase_dt, axis=1)
ddphase_df2 = np.diff(dphase_df, axis=0)

# Расчет непрерывности фазы (phase continuity)
phase_continuity = (gammatone_phase_spectrogram[:, :-1] - gammatone_phase_spectrogram[:, 1:]) % (2 * np.pi)
