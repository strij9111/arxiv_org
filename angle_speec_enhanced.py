"""
используя азимутальную информацию о звуковом источнике, усилить сигнал от диктора и подавить фоновый шум
алгоритм задержки и суммирования (Delay-and-Sum Beamforming).

Предположим, что вы уже получили азимутальный угол звукового источника, используя функцию estimate_azimuth_angle, и имеете сигналы от микрофонного массива. Вам потребуется выполнить следующие шаги:

1. Рассчитать задержку времени для каждого микрофона в массиве на основе азимутального угла и расстояния между микрофонами.
2. Задержать сигналы от каждого микрофона в массиве на соответствующие рассчитанные временные задержки. Это создаст фазовый сдвиг сигналов в пространстве таким образом, что сигнал от диктора будет выстроен в фазе, а шум с других направлений будет разнесен в фазе.
3. Суммировать задержанные сигналы от всех микрофонов для получения усиленного сигнала от диктора. Так как сигналы от диктора теперь выстроены в фазе, их суммирование приведет к усилению сигнала, в то время как некоррелированный фоновый шум будет подавлен.
4. При необходимости применить дополнительные алгоритмы обработки сигналов, такие как фильтрация или улучшение речи, к полученному суммарному сигналу.
"""

import numpy as np

def delay_and_sum(mic_signals, azimuth_angle, mic_distances, speed_of_sound=343):
    num_mics = len(mic_signals)
    num_samples = len(mic_signals[0])

    # Рассчитать временные задержки для каждого микрофона
    time_delays = mic_distances / speed_of_sound * np.sin(np.deg2rad(azimuth_angle))

    # Задержать сигналы от каждого микрофона
    delayed_signals = []
    for i, signal in enumerate(mic_signals):
        delay_samples = int(time_delays[i] * sample_rate)
        delayed_signal = np.roll(signal, delay_samples)
        delayed_signals.append(delayed_signal)

    # Суммировать задержанные сигналы от всех микрофонов
    enhanced_signal = np.sum(delayed_signals, axis=0) / num_mics

    return enhanced_signal
