import numpy as np

# Функция решения задачи квадратного программирования
def quadratic_programming_problem(A, d, x, J_op, J, B, debug=False):
    # Вычисление матрицы D
    D = np.dot(B.T, B)
    # Вычисление вектора c
    c = - np.dot(d.T, B)
    # Преобразование индексов J и J_op из единицы в ноль
    J -= 1
    J_op -= 1
    m, n = A.shape
    iteration = 0
    skip_1 = False
    j0 = 0

    # Основной цикл итераций
    while True:
        iteration += 1

        print('==================================================================================')
        if not skip_1:
            # Вычисление вспомогательных переменных
            not_J_op = np.delete(np.arange(n), J_op)
            c_x = c + np.dot(x, D)
            A_op_inv = np.linalg.inv(A[:, J_op])
            u_x = - c_x[J_op].dot(A_op_inv)
            delta = u_x.dot(A) + c_x
            round_delta = np.array(list(map(lambda _x: round(float(_x), 4), delta)))

            # Проверка оптимальности текущего плана
            if (round_delta >= 0).all():
                print()
                print(list(map(lambda _x: round(float(_x), 3), list(x))), " - оптимальный план")
                print("max = ", round(c.dot(x) + 0.5 * x.dot(D).dot(x), 4))
                print("Количество итераций: ", iteration)
                return x.tolist(), c.dot(x) + 0.5 * x.dot(D).dot(x), iteration

            else:
                ind = np.argmin(delta[not_J_op])
                j0 = not_J_op[ind]

        skip_1 = False
        l = np.zeros(n)
        l[j0] = 1

        len_J = len(J)
        H = np.zeros((m + len_J, m + len_J))
        H[:len_J, :len_J], H[len_J:, :len_J], H[:len_J, len_J:] = D[J][:, J], A[:, J], A[:, J].T

        b_z = np.zeros(m + len_J)
        b_z[: len_J], b_z[len_J:] = D[j0][J], A[:, j0]

        # Решение системы линейных уравнений
        x_kr = np.linalg.inv(H).dot(-b_z)
        l[J] = x_kr[: len_J]

        # Вычисление тета и выбор базисной переменной
        theta = [-x[i] / l[i] if l[i] < 0 else np.inf for i in J]
        j_z = np.argmin(theta)

        small_delta = l.T.dot(D).dot(l)
        theta_j0 = np.inf if 0 < abs(small_delta) <= 1.0e-10 else np.abs(delta[j0] / small_delta)
        theta_0, j_z = (theta_j0, j0) if theta[j_z] >= theta_j0 else (theta[j_z], J[j_z])

        if debug:
            print('*****************************')
            print("Итерация: %d" % iteration)
            print("H:")
            print(*H, sep='\n')
            print("l = ", *l)
            print("j0 = ", j0)
            print("j* = ", j_z)
            print("theta_0 = ", theta_0)
            print("delta = ", *delta)
            print('*****************************')

            print()

        if np.isinf(theta_0):
            print("Нет решения, так как целевая функция не ограничена на множестве допустимых планов")
            return
        x = x + theta_0 * l
        if debug:
            print("Новый план: ", x)

        if j0 == j_z:  # case 1
            J = np.append(J, j_z)

        elif j_z in set(J) - set(J_op):  # case 2
            J = J[J != j_z]
            delta[j0] += theta_0 * small_delta
            skip_1 = True
            J_op.sort()
            J.sort()

        else:
            s = np.where(J_op == j_z)
            e_s = np.eye(m)[s]
            j_plus = set(J) - set(J_op)

            print("J_plus", j_plus)

            t = list(filter(lambda i: e_s.dot(A_op_inv).dot(A[:, i]) != 0, j_plus))
            print("t: ", t)

            if t:  # case 3
                J_op = np.append(J_op[J_op != j_z], int(t[0]))
                J = J[J != j_z]
                delta[j0] += theta_0 * small_delta
                skip_1 = True
                J_op.sort()
                J.sort()

            else:  # case 4
                J_op = np.append(J_op[J_op != j_z], j0)
                J = np.append(J[J != j_z], j0)

        if debug:
            print("J_op = ", J_op)
            print("J* = ", J)


# Тесты

def test_1():
    A = np.array([
        [1, 2, 0, 1, 0, 4, -1, -3],
        [1, 3, 0, 0, 1, -1, -1, 2],
        [1, 4, 1, 0, 0, 2, -2, 0]
    ])
    b = np.array([4, 5, 6])
    B = np.array([
        [1, 1, -1, 0, 3, 4, -2, 1],
        [2, 6, 0, 0, 1, -5, 0, -1],
        [-1, 2, 0, 0, -1, 1, 1, 1]
    ])
    d = np.array([7, 3, 3])
    x = np.array([0, 0, 6, 4, 5, 0, 0, 0])
    J_op = np.array([3, 4, 5])
    J = np.array([3, 4, 5])

    quadratic_programming_problem(A, d, x, J_op, J, B, True)


def test_2():
    A = np.array([
        [11, 0, 0, 1, 0, -4, -1, 1],
        [1, 1, 0, 0, 1, -1, -1, 1],
        [1, 1, 1, 0, 1, 2, -2, 1]
    ])

    B = np.array([
        [1, -1, 0, 3, -1, 5, -2, 1],
        [2, 5, 0, 0, -1, 4, 0, 0],
        [-1, 3, 0, 5, 4, -1, -2, 1]
    ])
    d = np.array([6, 10, 9])
    x = np.array([0.7273, 1.2727, 3, 0, 0, 0, 0, 0])
    J_op = np.array([1, 2, 3])
    J = np.array([1, 2, 3])

    quadratic_programming_problem(A, d, x, J_op, J, B, True)


if __name__ == "__main__":
    test_1()
    input()

    test_2()
