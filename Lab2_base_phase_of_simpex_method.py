import numpy as np


def simplex_main_phase(c, A, x, B):
    iteration = 1
    while True:
        print(f"Iteration {iteration}")
        m, n = A.shape

        # Шаг 1: Вычисление обратной матрицы для базисных переменных
        B_inv = np.linalg.inv(A[:, B])

        # Шаг 2: Формирование вектора cB
        c_B = c[B]

        # Шаг 3: Находим вектор потенциалов u
        u = c_B @ B_inv
        print("u:", u)

        # Шаг 4: Находим вектор оценок ∆
        delta = u @ A - c
        print("delta:", delta)

        # Шаг 5: Проверка условия оптимальности
        if np.all(delta >= 0):
            print("Оптимальное решение найдено!")
            return x

        # Шаг 6: Находим первую отрицательную компоненту в ∆
        j0 = np.argmin(delta)
        print("j0:", j0)

        # Шаг 7: Вычисляем вектор z
        Aj0 = A[:, B] @ A[:, j0]
        z = B_inv @ Aj0
        print("z:", z)

        # Шаг 8: Вычисляем вектор θ
        theta = np.full(m, np.inf)
        positive_indices = np.where(z > 0)[0]
        theta[positive_indices] = x[np.array(B).astype(int)[positive_indices]] / z[positive_indices]
        print("theta:", theta)

        # Шаг 9: Вычисляем θ0
        theta_0 = np.min(theta)
        print("theta_0:", theta_0)

        # Шаг 10: Проверка условия неограниченности
        if theta_0 == np.inf:
            print("Целевая функция не ограничена сверху на множестве допустимых планов")
            return "Целевая функция не ограничена сверху на множестве допустимых планов"

        # Шаг 11: Находим индекс k, на котором достигается минимум в θ
        k = np.argmin(theta)
        print("k:", k)

        # Шаг 12: Обновление множества B
        B[k] = j0
        print("B:", B)

        # Шаг 13: Обновление компонент плана x
        x = x.astype(np.float64)  # Приведение к типу с плавающей точкой
        x_B = x[B]
        x_B -= theta_0 * z
        x[B] = x_B
        x[B[k]] = theta_0
        x[np.setdiff1d(range(n), B)] = 0
        print("x:", x)

        iteration += 1



# Пример 1 (методичка)


# # Входные данные
# c = np.array([1, 1, 0, 0, 0])
# A = np.array([[-1, 1, 1, 0, 0],
#               [1, 0, 0, 1, 0],
#               [0, 1, 0, 0, 1]])
# x = np.array([0, 0, 1, 3, 2])
# # B = np.array([2, 3, 4])  # Индексы базисных переменных
# #
# # # Вызов функции
# optimal_solution = simplex_main_phase(c, A, x, B)
# print("Оптимальное решение:", optimal_solution.astype(int))


# # Пример 2
#
# # Входные данные
# c = np.array([1, 0, 0, 0, 0])  # Задаем целевую функцию
# A = np.array([[1, -1, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, -1, 0]])  # Определяем ограничения
# x = np.array([0, 0, 0, 0, 0])  # Начальное допустимое решение
# B = np.array([0, 1, 2])  # Индексы базисных переменных

# # Вызываем функцию
# optimal_solution = simplex_main_phase(c, A, x, B)
# if isinstance(optimal_solution, str):
#     print(optimal_solution)  # Если решение - строка, выводим ее
# else:
#     print("Оптимальное решение:", optimal_solution.astype(int))