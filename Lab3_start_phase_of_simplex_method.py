from Lab2_base_phase_of_simpex_method import simplex_main_phase
import numpy as np


def initial_phase_simplex(c, A, b, simplex_main_phase):
    m, n = A.shape

    # Создаем вспомогательную задачу с искусственными переменными
    c_aux = np.zeros(n + m)  # целевая функция для вспомогательной задачи
    c_aux[-m:] = -1  # коэффициенты искусственных переменных
    A_aux = np.hstack((A, np.eye(m)))  # матрица коэффициентов для вспомогательной задачи
    x_bounds = [(0, None) for _ in range(n)] + [(0, None) for _ in range(m)]  # границы для x и y

    # Решаем вспомогательную задачу симплекс-методом
    x_initial = simplex_main_phase(c_aux, A_aux, np.zeros(n + m), list(range(n, n + m)))

    if isinstance(x_initial, str):
        print("Начальное базисное решение не найдено.")
        return None

    # Получаем допустимое базисное решение из результата вспомогательной задачи
    x_B_initial = x_initial[:n]

    return x_B_initial


# Входные данные
c = np.array([0] * 3)
A = np.array([[1, 1, 1], [2, 2, 2]])
b = np.array([0, 0])

# Пример использования:
x_initial = initial_phase_simplex(c, A, b, simplex_main_phase)
print("Начальное базисное решение x^T =", x_initial)
