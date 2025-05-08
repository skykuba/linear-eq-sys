import numpy as np
import matplotlib.pyplot as plt

class GaussSeidel:
    def __init__(self, A, b, x0=None, eps=1e-9, max_iter=1000):
        self.A = A
        self.b = b
        self.x = x0.copy() if x0 is not None else np.zeros_like(b)
        self.eps = eps
        self.max_iter = max_iter
        self.N = len(b)
        self._validate_inputs()

    def _validate_inputs(self):
        if self.A.shape != (self.N, self.N):
            raise ValueError("Macierz A musi być kwadratowa i pasować do rozmiaru wektora b")
        if np.any(np.diag(self.A) == 0):
            raise ValueError("Macierz A ma zero na przekątnej - dzielenie przez zero")

    def calculate(self):
        residuum_norms = []
        x_prev = self.x.copy()

        # Oblicz początkowy residuum
        residual = self.A @ self.x - self.b
        current_norm = np.linalg.norm(residual)
        residuum_norms.append(current_norm)
        iter = 0

        # Główna pętla algorytmu
        while current_norm > self.eps and iter < self.max_iter:
            # Aktualizacja współrzędnych x w miejscu
            for i in range(self.N):
                sigma = np.dot(self.A[i, :i], self.x[:i]) + np.dot(self.A[i, i + 1:], x_prev[i + 1:])
                self.x[i] = (self.b[i] - sigma) / self.A[i, i]

            # Oblicz nowy residuum
            residual = self.A @ self.x - self.b
            current_norm = np.linalg.norm(residual)
            residuum_norms.append(current_norm)

            # Przygotuj x_prev dla następnej iteracji
            x_prev[:] = self.x.copy()

            iter += 1
        self.residuum=residuum_norms
        return self.x, residuum_norms, iter

    def show_residuum(self):
        plt.plot(self.residuum, marker='o')
        plt.yscale('log')
        plt.title('Norma residuum w trakcie iteracji Gaus')
        plt.xlabel('Iteracja')
        plt.ylabel('Norma residuum')
        plt.grid()
        plt.show()