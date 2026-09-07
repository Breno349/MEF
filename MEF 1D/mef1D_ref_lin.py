import numpy as np
import matplotlib.pyplot as plt

#=========  Criando domínio  =========
n = 31                 # Número de nós do domínio computacional
n_grid = 500;          # Número de nós do grid
cd_ns = [0]             # Nós sob condição de contorno de Diriclhet
cd_vs = [0]             # Valores nesses nós
cn_ns = [n-1]             # Nós sob condição de contorno de Newmann
cn_vs = [50]             # Valores nesses nós

coords = np.linspace(0,1,n)                 # Vetor de coordenadas | Coordenadas dos nós | Domínio
elements = [(j-1,j) for j in range(1,n)]    # Matriz de conectividade | Matriz de nós | Elementos

# Função analitica do problema
def an(x):
    return (-(x**2)/2 + 6*x)*1e-8

# Fatores {constantes} dos elementos
EA = 1e9          # Fator que multiplica a matriz [Ke]
f = 10           # Fator que multiplica o vetor [Fe]

#=========  Alocando as matrizes =========
K = np.zeros((n,n))
F = np.zeros((n,1))

#=========  Fazendo cálculo de cada matriz elementar e realizando assemblagem  =========
for no1,no2 in elements:
    x1,x2 = coords[[no1,no2]]
    
    l = (x2 - x1)
    Ke = [
        [1/l, -1/l],
        [-1/l, 1/l]
    ]
    Fe = [
        l/2,
        l/2
    ];

    #=========  Assemblagem  =========
    nos_p = [no1,no2]
    for i in range(2):
        F[nos_p[i], 0] += f*Fe[i]
        for j in range(2):
            K[nos_p[i], nos_p[j]] += EA*Ke[i][j]

#=========  Condições de contorno de Diriclhet  =========
for i in range(len(cd_ns)):
    cd_n = cd_ns[i]
    cd_v = cd_vs[i]
    F[:, 0] -= K[:, cd_n]*cd_v
    K[:, cd_n] = 0
    K[cd_n, :] = 0
    K[cd_n, cd_n] = 1
    F[cd_n, 0] = cd_v
#=========  Condições de contorno de Newmann  =========
for i in range(len(cn_ns)):
    cn_n = cn_ns[i]
    cn_v = cn_vs[i]
    F[cn_n, 0] += cn_v

#=========  Resolver o sistema  =========
ux = np.linalg.solve(K, F).flatten()

#=========  Função de erro  =========
def erro(apx, ref):
    err = sum((ref - apx)**2)
    err = err/sum((ref)**2)
    return np.sqrt(err)

#=========  Interpolação dos dados  =========
x_grid = np.linspace(0,1,n_grid)
u_grid = np.zeros(n_grid)
a_grid = an(x_grid)
tol = 1e-9

for i,xg in enumerate(x_grid):
    for no1,no2 in elements:
        x1,x2 = coords[[no1,no2]]
        if x1 <= xg <= x2 + tol:

            N1 = (x2-xg)/(x2-x1)
            N2 = (xg-x1)/(x2-x1)

            u_1 = ux[no1]
            u_2 = ux[no2]

            u_val = N1*u_1 + N2*u_2

            u_grid[i] = u_val
            break

#=========  Calculando erro  =========
err = erro(u_grid, a_grid)
print("Erro:", err)

#=========  Plotando resultados  =========
plt.plot(x_grid, a_grid)
plt.plot(x_grid, u_grid, '--')
plt.grid()
plt.legend(["Analitica","Aproximada"])
plt.show()