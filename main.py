import numpy as np
import time
from jacobi import Jacobi
from gauss_seidel import GaussSeidel
INDEX_NUMBER='198088'

N=12000+10*int(INDEX_NUMBER[-2])+int(INDEX_NUMBER[-1])
RESIDUUM_BREAK_POINT=10**(-6)
def createA(a1, a2, a3,N):
    a = np.zeros((N, N))
    np.fill_diagonal(a, a1)  # Główna przekątna
    np.fill_diagonal(a[1:], a2)  # Przekątna poniżej głównej
    np.fill_diagonal(a[:, 1:], a2)  # Przekątna powyżej głównej
    np.fill_diagonal(a[2:], a3)  # Druga przekątna poniżej głównej
    np.fill_diagonal(a[:, 2:], a3)  # Druga przekątna powyżej głównej
    return a

def createB(N,f=INDEX_NUMBER[2]):
    b = np.zeros((N,))
    for i in range(N):
        b[i] = np.sin((i + 1) * (int(f) + 1))
    return b

def residuum(a,x,b):
    return np.dot(a,x)-b


def exerciseA():
    a1=5+int(INDEX_NUMBER[3])
    a2=a3=-1
    a=createA(a1,a2,a3,N)
    b=createB(N)
    return a,b

def exerciseA_test():
    n=5
    a1=5+int(INDEX_NUMBER[3])
    a2=a3=-1
    a=createA(a1,a2,a3,n)
    b=createB(n)
    return a,b

def calculate_Jacobi(a,b):
    jacobi = Jacobi(a, b)
    x, inorm,iterations = jacobi.calculate()
    return iterations, x, inorm,jacobi

def calculate_GaussSeidel(a,b):
    gauss_seidel = GaussSeidel(a, b)
    x, inorm,iterations = gauss_seidel.calculate()
    return iterations, x, inorm,gauss_seidel

def calculate_brute_force(a,b):
    x = np.linalg.solve(a, b)
    inorm = np.linalg.norm(residuum(a,x,b), ord=np.inf)
    return x, inorm

def exerciseB(test=False):
    A,b=exerciseA_test() if test else exerciseA()

    print(f"shape A {A.shape} shape b {b.shape}")
    start = time.time()
    iterations, x, inorm,jacobi = calculate_Jacobi(A, b)
    delta = time.time() - start
    print("Jacobi")
    print(iterations)
    print(x)
    print(inorm[-1])
    print()
    print("czas ", delta)
    jacobi.show_residuum()


    start = time.time()
    iterations, x, inorm ,gauss_seidel= calculate_GaussSeidel(A, b)
    delta = time.time() - start
    print("Gauss-Seidel")
    print(iterations)
    print(x)
    print(inorm[-1])
    print()
    print("czas ", delta)
    gauss_seidel.show_residuum()
    start = time.time()
    x, inorm = calculate_brute_force(A, b)
    delta = time.time() - start
    print("Brute force")
    print(x)
    print(inorm)
    print("czas ", delta)

def exerciseC():
    a1=3
    a2=a3=-1
    A=createA(a1,a2,a3,N)
    b=createB(N)
    start=time.time()
    iterations,x,inorm,jacobi=calculate_Jacobi(A,b)
    delta=time.time()-start
    print("Jacobi")
    print(iterations)
    print(x)
    print(inorm[-1])
    print()
    print("czas ", delta)
    jacobi.show_residuum()
    start = time.time()
    iterations, x, inorm, gauss_seidel = calculate_GaussSeidel(A, b)
    delta = time.time() - start
    print("Gauss-Seidel")
    print(iterations)
    print(x)
    print(inorm[-1])
    print()
    print("czas ", delta)
    gauss_seidel.show_residuum()

def exerciseD():
    A,b=exerciseA()

exerciseC()