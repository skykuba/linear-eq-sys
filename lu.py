import copy
import numpy as np


class LU:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.N = A.shape[0]



    def lu_decomposition2(self):
        N = self.A.shape[0]
        L = np.eye(N)
        U = np.zeros((N, N))

        for i in range(N):
            # Obliczanie elementów macierzy U
            U[i, i:] = self.A[i, i:] - L[i, :i] @ U[:i, i:]
            # Obliczanie elementów macierzy L
            L[i + 1:, i] = (self.A[i + 1:, i] - L[i + 1:, :i] @ U[:i, i]) / U[i, i]

        return L, U

    def lu_decomposition(self):
        U = self.A.copy()
        L = np.eye(self.N)

        for i in range(self.N):
            if U[i, i] == 0:
                raise ValueError("Znaleziono zerowy element na przekątnej. Algorytm wymaga pivotingu.")

            for j in range(i + 1, self.N):
                L[j, i] = U[j, i] / U[i, i]
                U[j, :] -= L[j, i] * U[i, :]


        return L,U

    def solve2(self):
        L, U = self.lu_decomposition()

        # Rozwiązywanie L * y = b
        y = np.linalg.solve(L, self.b)
        x = np.linalg.solve(U, y)

        return x