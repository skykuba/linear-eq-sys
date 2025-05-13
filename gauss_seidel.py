import numpy as np
import matplotlib.pyplot as plt

class GaussSeidel:
    def __init__(self, A, b, x0=None, eps=1e-9, max_iter=1000):
        self.A = A
        self.b = b
        self.x0 = x0.copy() if x0 is not None else np.zeros(len(b))
        self.eps = eps
        self.max_iter = max_iter
        self.N = len(b)
        self._validate_inputs()
        self.divergence_threshold=10**9

    def _validate_inputs(self):
        if self.A.shape != (self.N, self.N):
            raise ValueError("Macierz A musi być kwadratowa i pasować do rozmiaru wektora b")
        if np.any(np.diag(self.A) == 0):
            raise ValueError("Macierz A ma zero na przekątnej - dzielenie przez zero")

    def calculate(self):
        residuum_norms = []
        self.x = self.x0
        residual = self.A @ self.x - self.b
        inorm = np.linalg.norm(residual)
        residuum_norms.append(inorm)
        iter = 0

        while inorm > self.eps and iter < self.max_iter:
            for i in range(self.N):
                sigma = np.dot(self.A[i, :i], self.x[:i]) + np.dot(self.A[i, i + 1:], self.x[i + 1:])
                self.x[i] = (self.b[i] - sigma) / self.A[i, i]

            residual = self.A @ self.x - self.b
            inorm = np.linalg.norm(residual)
            residuum_norms.append(inorm)

            iter += 1
            if inorm > self.divergence_threshold:
                break

        self.residuum = residuum_norms
        return self.x, residuum_norms, iter
    def show_residuum(self):
        plt.plot(self.residuum, marker='o')
        plt.yscale('log')
        plt.title('Norma residuum w trakcie iteracji Gaus')
        plt.xlabel('Iteracja')
        plt.ylabel('Norma residuum')
        plt.grid()
        plt.show()