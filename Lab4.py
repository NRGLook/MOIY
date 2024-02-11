from numpy import *


# c - вектор коэф цел функции
# b - вектор правых частей ограничений\
# a_matrix - матрица коэф ограничений
# j_vector - вектор индексов базисных переменных

def double_simplex(c, b, a_matrix, j_vector):
    # Получаем количество строк и столбцов матрицы a_matrix
    m, n = a_matrix.shape
    # Инициализируем счетчик итераций
    iter_count = 1
    # Преобразуем индексы вектора j_vector к нулевой базе
    j_vector -= 1
    # Получаем начальное значение вектора y
    y = get_initial_y(c, a_matrix, j_vector)
    # Инициализируем вектор x_0 нулями
    x_0 = [0 for _ in range(n)]

    while True:
        # Создаем вектор, содержащий индексы переменных, не входящих в базис
        not_J = delete(arange(n), j_vector)
        # Вычисляем обратную матрицу базисных столбцов
        B = linalg.inv(a_matrix[:, j_vector])
        # Вычисляем значения переменных базиса
        kappa = B.dot(b)

        # Проверяем условие оптимальности
        if all(kappa >= 0):
            # Если все компоненты kappa >= 0, то достигнуто оптимальное решение
            for j, _kappa in zip(j_vector, kappa):
                x_0[j] = _kappa

            print("Количество итераций : ", iter_count)
            print("Максимальная прибыль : ", c.dot(x_0))
            print(list(map(lambda _x: round(float(_x), 3), list(x_0))), "-  план")

            return x_0, iter_count

        # Выбираем индекс входящей переменной
        k = argmin(kappa)
        # Вычисляем вектор дельта y
        delta_y = B[k]
        # Вычисляем вектор mu
        mu = delta_y.dot(a_matrix)

        # Выводим вектор mu и текущий вектор y
        print("mu: \n\t", mu)
        print("y: \n\t", y)

        # Вычисляем вектор sigma
        sigma = []
        for i in not_J:
            if mu[i] >= 0:
                sigma.append(inf)
            else:
                sigma.append((c[i] - a_matrix[:, i].dot(y)) / mu[i])

        # Выводим вектор sigma
        print("sigma:\n\t", sigma)

        # Находим индекс ведущей переменной и соответствующее значение sigma_0
        sigma_0_ind = not_J[argmin(sigma)]
        sigma_0 = min(sigma)

        # Выводим индекс и значение sigma_0
        print("Sigma: \n\tval: {0} \n\tindex: {1}".format(sigma_0, sigma_0_ind))

        # Проверяем, имеется ли допустимое решение
        if sigma_0 == inf:
            print("Задача не имеет решения, т.к. пусто множество ее допустимых планов.")
            return "Задача не имеет решения"

        # Обновляем вектор y и вектор j_vector
        y += sigma_0 * delta_y
        j_vector[k] = sigma_0_ind
        iter_count += 1

# Функция для вычисления начального значения вектора y
def get_initial_y(c, a_matrix, j_vector):
    return (c[j_vector]).dot(linalg.inv(a_matrix[:, j_vector]))

# Функция для тестирования метода на примере test_1_small
def test_1_small():
    A = array([
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1]
    ])
    b = array([-1, -1.5])
    c = array([-4, -3, -7, 0, 0])
    J = array([4, 5])

    double_simplex(c, b, A, J)

# Функция для тестирования метода на примере example_1
def example_1():
    A = array([
        [-2, -1, 1, -7, 0, 0, 0, 2],
        [4, 2, 1, 0, 1, 5, -1, -5],
        [1, 1, 0, -1, 0, 3, -1, 1]
    ])
    b = array([-2, 4, 3])
    c = array([2, 2, 1, -10, 1, 4, -2, -3])
    J = array([2, 5, 7])

    double_simplex(c, b, A, J)

# Функция для тестирования метода на примере test_3
def test_3():
    A = array([
        [-2, -1, 1, -7, 0, 0, 0, 2],
        [-4, 2, 1, 0, 1, 5, -1, 5],
        [1, 1, 0, 1, 4, 3, 1, 1]
    ])
    b = array([-2, 8, -2])
    c = array([12, -2, -6, 20, -18, -5, -7, -20])
    J = array([2, 4, 6])

    double_simplex(c, b, A, J)

# Функция для тестирования метода на примере test_4
def test_4():
    A = array([
        [-2, -1, 10, -7, 1, 0, 0, 2],
        [-4, 2, 3, 0, 5, 1, -1, 0],
        [1, 1, 0, 1, -4, 3, -1, 1]
    ])
    b = array([-2, -5, 2])
    c = array([10, -2, -38, 16, -9, -9, -5, -7])
    J = array([2, 8, 5])

    double_simplex(c, b, A, J)

# Функция для тестирования метода на примере test_5
def test_5():
    A = array([
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1],
    ])
    b = array([-1, -3/2])
    c = array([-4, -3, -7, 0, 0])
    J = array([4, 5])

    double_simplex(c, b, A, J)

# Основная функция для запуска тестов
def main():
    test_1_small()
    input()

    example_1()
    input()

    test_3()
    input()

    test_4()
    input()

    test_5()

if __name__ == "__main__":
    main()
