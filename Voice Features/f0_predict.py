"""
использование линейной регрессии для предсказания F0 на основе крупных спектральных характеристик
"""
import librosa
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def extract_f0(audio_file):
    y, sr = librosa.load(audio_file)
    f0_series = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    return f0_series

def load_data(audio_files):
    X = np.vstack([extract_features(f) for f in audio_files])
    y = np.hstack([extract_f0(f) for f in audio_files])
    return X, y

audio_files = [...]  # список аудиофайлов для анализа
X, y = load_data(audio_files)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R2: {r2}')
