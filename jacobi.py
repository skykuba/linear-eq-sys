import numpy as np
import matplotlib.pyplot as plt


class Jacobi:
    def __init__(self,A,b,x0=None,eps=10**(-9),max_iter=1000):
        self.A = A
        self.b = b
        self.x0 = x0 if x0 is not None else np.zeros(len(b))
        self.eps = eps
        self.max_iter = max_iter
        self.N= len(b)
        self.divergence_threshold=10**9

    def calculate(self):
        D = np.diag(self.A)
        L_plus_U = self.A - np.diag(D)
        self.x=self.x0
        residuum_norms=[]
        residual=self.A.dot(self.x)-self.b
        current_norm = np.linalg.norm(residual)
        residuum_norms.append(current_norm)
        inorm=np.inf
        iter=0
        while inorm>self.eps and iter<self.max_iter:
            self.x=(self.b - L_plus_U.dot(self.x)) / D
            inorm=np.linalg.norm(self.A.dot(self.x)-self.b)
            residuum_norms.append(inorm)
            iter+=1
            if inorm>self. divergence_threshold:
                break
        self.residuum=residuum_norms
        return self.x,residuum_norms,iter

    def show_residuum(self):
        plt.plot(self.residuum, marker='o')
        plt.yscale('log')
        plt.title('Norma residuum w trakcie iteracji Jacobi')
        plt.xlabel('Iteracja')
        plt.ylabel('Norma residuum')
        plt.grid()
        plt.show()