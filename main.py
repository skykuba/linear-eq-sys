import numpy as np

INDEX_NUMBER='198088'

N=1200+10*int(INDEX_NUMBER[-2])+int(INDEX_NUMBER[-1])
RESIDUUM_BREAK_POINT=10**(-6)
def createA(a1, a2, a3):
    a = np.zeros((N, N))
    np.fill_diagonal(a, a1)  # Główna przekątna
    np.fill_diagonal(a[1:], a2)  # Przekątna poniżej głównej
    np.fill_diagonal(a[:, 1:], a2)  # Przekątna powyżej głównej
    np.fill_diagonal(a[2:], a3)  # Druga przekątna poniżej głównej
    np.fill_diagonal(a[:, 2:], a3)  # Druga przekątna powyżej głównej
    return a

def residuum(a,x,b):
    return np.dot(a,x)-b


def exerciseA():
    a1=5+int(INDEX_NUMBER[3])
    a2=a3=-1
    a=createA(a1,a2,a3)
    b=np.zeros(N,)
    for i in range(N):
        b[i]=np.sin((i+1)*(int(INDEX_NUMBER[2])+1))

exerciseA()

