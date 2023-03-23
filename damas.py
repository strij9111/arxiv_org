def damas_algorithm(B, CSDM, n_iter=100):
    n_sources = B.shape[1]
    n_freqs = B.shape[0]

    # Нормализация матрицы пропагации
    B_norm = B / np.linalg.norm(B, axis=0)

    # Предварительное вычисление матрицы пропагации передаточных функций
    BB = np.matmul(B_norm.conj().T, B_norm)

    # Инициализация вектора источников
    source_powers = np.ones(n_sources) / n_sources

    # Итерационный процесс деконволюции
    for _ in range(n_iter):
        B_source_powers = np.matmul(B_norm, source_powers)
        CSDM_reconstructed = np.abs(B_source_powers[:, None] * B_source_powers[None, :]) ** 2
        CSDM_ratios = np.divide(CSDM, CSDM_reconstructed, out=np.ones_like(CSDM), where=CSDM_reconstructed != 0)
        B_ratios = np.matmul(B_norm.conj().T, CSDM_ratios)
        source_powers *= np.diag(B_ratios).real
        source_powers /= np.sum(source_powers)

    return source_powers

# Пример применения алгоритма DAMAS
# Здесь нужно предоставить корректные значения для матрицы B (пропагации передаточных функций) и CSDM (кросс-спектральной матрицы)
# В реальном сценарии эти значения будут зависеть от конкретной задачи и расположения микрофонов и источников шума

# Примерные размеры матриц:
n_freqs = B.shape[0]
n_sources = B.shape[1]

# Создание случайной матрицы B для демонстрации
B_demo = np.random.randn(n_freqs, n_sources) + 1j * np.random.randn(n_freqs, n_sources)

# Создание случайной кросс-спектральной матрицы для демонстрации
CSDM_demo = np.random.rand(n_freqs, n_freqs)

# Применение алгоритма DAMAS
source_powers_demo = damas_algorithm(B_demo, CSDM_demo)
