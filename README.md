# linear-eq-sys

**linear-eq-sys** to prosty moduł Pythona do rozwiązywania układów równań liniowych. Przydatny zarówno w celach edukacyjnych, jak i praktycznych.

## Wymagania

- Python

## Instalacja

```bash
git clone https://github.com/skykuba/linear-eq-sys.git
cd linear-eq-sys
pip install -r requirements.txt
```

## Przykład użycia

```python
import numpy as np
from jacobi import Jacobi

A = np.array([[4, -1, 0, 0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 3]])
b = np.array([15, 10, 10, 10])

# Inicjalizacja klasy Jacobi
jacobi_solver = Jacobi(A, b)

# Obliczenia
solution, residuum_norms, iterations = jacobi_solver.calculate()
```

## Wkład i licencja

Zgłoszenia błędów i pull requesty są mile widziane.  
Licencja: MIT.
