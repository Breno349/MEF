import numpy as np
from scipy.integrate import quad as itg
import matplotlib.pyplot as plt

#=========================================================
#=========================================================
#   MEF 1D - Resolve problemas 1D simples via integração numerica, usa elementos lineares de 2 nós.
#   Resolve problemas do tipo: EA/dw*du = /f(x)*w(x) + w(L)*P   [0,L]
#=========================================================
#=========================================================

#===============  O que é editável  ===============
#       n      =>    número de nós
#       f(x)   =>    função f
#       u_n    =>    indice do nó com condição de contorno de dirichlet (C.C.D)
#       u_x    =>    valor desse nó
#       n_n    =>    indice do nó com condição de contorno de newmann (C.C.N)
#       n_x    =>    valor desse nó
#       an(x)  =>    função analitica do problema


#===============  Criando o domínio computacional 1D  ===============
n = 10                                           # Número de nós
coords = np.linspace(0,1,n)                     # Coordenadas dos nós | Matriz de nós
elements = [(j-1,j) for j in range(1,n)]        # Matriz de elementos | Matriz de conectividade

def f(x):
    return 10
def an(x):
    return (-(x**2)/2 + 6*x)*1e-8

u_n = 0                                         # Nó que tem condições de dirichlet
u_x = 0                                         # Valor desse nó
n_n = (n-1)                                     # Nó que tem condições de newmann
n_x = 50                                        # Valor da condição nesse nó
EA = 1e9

#===============  Alocando a Matriz de Rígidez e Vetor de Força nodal  ===============
K = np.zeros((n,n))                             # Relaciona as deformações entre os nós | Matriz de rigidez
F = np.zeros((n,1))                             # O resultado de cada nós numa força | Vetor de Força Nodal

#===============  Cálculando e montando as matrizes  ===============
for no1,no2 in elements:
    x1,x2 = coords[[no1,no2]]                   # Coordenadas de cada nó do elemento do looping
    N1 = lambda x: (x2-x)/(x2-x1)
    N2 = lambda x: (x-x1)/(x2-x1)
    dN1 = -1/(x2-x1)
    dN2 = 1/(x2-x1)
    Ke = [
        [
            itg(lambda x: (dN1*dN1), x1, x2)[0],
            itg(lambda x: (dN1*dN2), x1, x2)[0]
        ],
        [
            itg(lambda x: (dN2*dN1), x1, x2)[0],
            itg(lambda x: (dN2*dN2), x1, x2)[0]
        ],
    ]
    Fe = [
        itg(lambda x: f(x)*N1(x), x1, x2)[0],
        itg(lambda x: f(x)*N2(x), x1, x2)[0]
    ]
    #===============  Assemblagem  ===============
    nos_p = [no1, no2]
    for i in range(0,2):
        F[nos_p[i], 0] += Fe[i]
        for j in range(0,2):
            K[nos_p[i], nos_p[j]] += Ke[i][j]

K[:, :] *= EA

#===============  Condições de contorno de derichlet  ===============
# Famoso passar todo mundo para o outro lado
F[:, 0] -= K[:, u_n]*u_x
K[u_n, :] = 0
K[:, u_n] = 0

# Impondo o valor para C.C.D
K[u_n, u_n] = 1
F[u_n, 0] = u_x

# Impondo o valor para C.C.N
F[n_n, 0] += n_x


#===============  Resolver o sistema  ===============
u = np.linalg.solve(K, F)

#===============  Solução analítica  ===============
x = np.linspace(0,1,100)                    # Domínio para a analítica
a = an(x)                                   # Analitica

"""
plt.plot(x, a, lw=3.5)
plt.plot(coords, u, '--')
plt.grid()
plt.legend(["Analitica","Aproximada"])
plt.xlabel("x")
plt.ylabel("u(x)")
plt.show()
"""

#===============  Interpolação dos dados  ===============
n_grid = 500;
x_grid = np.linspace(0,1,n_grid)
u_grid = np.zeros(n_grid)
a_grid = an(x_grid)
tol = 1e-5

for i,xg in enumerate(x_grid):
    for no1,no2 in elements:
        x1,x2 = coords[[no1,no2]]
        N1 = (x2-xg)/(x2-x1)
        N2 = (xg-x1)/(x2-x1)

        if (np.abs(N1)+np.abs(N2) - 1) <= tol:
            u_0 = u[no1]
            u_1 = u[no2]
            ux = N1*u_0 + N2*u_1
            u_grid[i] = ux.item()
            break

#===============  Calculo de erro  ===============
def erro(apx, ref):
    err = sum((ref - apx)**2)
    err = err/sum((ref)**2)
    return np.sqrt(err)

err = erro(u_grid, a_grid)
print("Erro:", err)

#===============  Plotar resultados (analítica e solução) ambas avaliadas no grid  ===============
plt.plot(x_grid, a_grid)
plt.plot(x_grid, u_grid, '--r')
plt.grid()
plt.legend(["Analitica","Aproximada"])
plt.xlabel("x")
plt.ylabel("u(x)")
plt.show()