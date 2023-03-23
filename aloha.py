import numpy as np
import cvxpy as cp

def aloha(y, L, K):
    n = len(y)
    Y = np.zeros((L, n - L + 1))
    for i in range(L):
        Y[i] = y[i:i + n - L + 1]

    u, s, vh = np.linalg.svd(Y, full_matrices=False)
    V = vh.T

    x = cp.Variable(L)
    objective = cp.Minimize(cp.norm(V @ x))
    constraints = [cp.norm(x, 'inf') <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    x_opt = x.value
    f = np.roots(x_opt)
    f_abs = np.abs(f)

    return f_abs


# Использование ALOHA для восстановления спектра
y = np.random.randn(100)  # Пример измерений
L = 50  # Длина аннигилирующего фильтра
K = 10  # Количество ненулевых компонентов в спектре

# Применение ALOHA
f_abs = aloha(y, L, K)
print("Recovered spectrum components:", f_abs)

"""
Здесь мы реализовали функцию aloha, которая принимает на вход вектор измерений y, длину аннигилирующего фильтра L и количество ненулевых компонентов в спектре K. Функция возвращает восстановленные компоненты спектра f_abs. Обратите внимание,
"""