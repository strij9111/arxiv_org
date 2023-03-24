"""
Geneva Minimalistic Acoustic Parameter Set (eGeMAPS), and a combined feature set of eGeMAPS and VTL
"""
import librosa
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction


def extract_eGeMAPS_features(file_path):
    # Загрузите аудиофайл и извлеките его частоту дискретизации и данные
    [fs, x] = audioBasicIO.read_audio_file(file_path)

    # Вычисление eGeMAPS признаков с помощью pyAudioAnalysis
    eGeMAPS_features, _ = audioFeatureExtraction.stFeatureExtraction(x, fs, 0.050 * fs, 0.025 * fs)

    return eGeMAPS_features


def extract_VTL_features(file_path):
    # Загрузите аудиофайл и извлеките его частоту дискретизации и данные
    y, sr = librosa.load(file_path)

    # Вычисление признаков VTL с использованием librosa
    f0_series = librosa.yin(y, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'))
    vtl_feature = np.mean(f0_series)

    return vtl_feature


def extract_combined_features(file_path):
    eGeMAPS_features = extract_eGeMAPS_features(file_path)
    vtl_feature = extract_VTL_features(file_path)

    # Комбинирование признаков eGeMAPS и VTL
    combined_features = np.concatenate((eGeMAPS_features, np.array([vtl_feature])))

    return combined_features


# Пример использования функций
file_path = "path/to/your/audio/file.wav"

eGeMAPS_features = extract_eGeMAPS_features(file_path)
print("eGeMAPS features:", eGeMAPS_features)

vtl_feature = extract_VTL_features(file_path)
print("VTL feature:", vtl_feature)

combined_features = extract_combined_features(file_path)
print("Combined features:", combined_features)
