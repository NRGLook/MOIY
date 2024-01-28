import numpy as np


class MatrixInversionSolver:
    def __init__(self):
        self.A = None
        self.A_inv = None
        self.n = None

    def check_input(self):
        if self.A is None or self.A_inv is None:
            raise ValueError("Матрицы A и A_inv должны быть инициализированы перед использованием")
        if self.A.shape != (self.n, self.n) or self.A_inv.shape != (self.n, self.n):
            raise ValueError("Матрицы A и A_inv должны быть квадратными одинакового размера")
        if not np.allclose(np.dot(self.A, self.A_inv), np.eye(self.n)):
            raise ValueError("Матрица A_inv не является обратной для матрицы A")

    def input_matrices(self):
        self.n = int(input("Введите размерность квадратной матрицы: "))
        print("Введите элементы матрицы A построчно, разделяя элементы пробелами:")
        self.A = np.array([list(map(float, input().split())) for _ in range(self.n)])
        print("Введите элементы обратной матрицы A_inv построчно, разделяя элементы пробелами:")
        self.A_inv = np.array([list(map(float, input().split())) for _ in range(self.n)])
        self.check_input()

    def step_1(self, x, i):
        ell = np.dot(self.A_inv, x)
        if ell[i] == 0:
            print("Матрица A необратима")
            return None
        return ell

    def step_2(self, ell, i):
        elle = np.copy(ell)
        elle[i] = -1
        return elle

    def step_3(self, ell, elle, i):
        ell_hat = -1 / ell[i] * elle
        return ell_hat

    def step_4(self, i, ell_hat):
        Q = np.eye(self.n)
        Q[:, i] = ell_hat
        return Q

    def step_5(self, Q):
        A_inv_new = np.dot(Q, self.A_inv)
        return A_inv_new

    def solve(self, x, i):
        ell = self.step_1(x, i)
        if ell is None:
            return None

        elle = self.step_2(ell, i)
        ell_hat = self.step_3(ell, elle, i)
        Q = self.step_4(i, ell_hat)
        A_inv_new = self.step_5(Q)

        return A_inv_new


solver = MatrixInversionSolver()
solver.input_matrices()

x = list(map(float, input("Введите элементы вектора x, разделяя их пробелами: ").split()))
if len(x) != solver.n:
    raise ValueError("Вектор x должен иметь длину, равную размерности матрицы A")

i = int(input("Введите индекс столбца для замены (отсчет с 1): ")) - 1

A_inv_new = solver.solve(x, i)
if A_inv_new is not None:
    print("Новая обратная матрица A:\n", A_inv_new)
