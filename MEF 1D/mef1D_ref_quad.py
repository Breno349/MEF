import numpy as np
import matplotlib.pyplot as plt

#=========  Criando domínio  =========
n = 31                  # Número de nós do domínio computacional
n_grid = 500;           # Número de nós do grid
cd_ns = [0]             # Nós sob condição de contorno de Diriclhet
cd_vs = [0]             # Valores nesses nós
cn_ns = [n-1]             # Nós sob condição de contorno de Newmann
cn_vs = [50]             # Valores nesses nós

coords = np.linspace(0,1,n)                         # Vetor de coordenadas | Coordenadas dos nós | Domínio
elements = [ (j-2,j-1,j) for j in range(2,n,2) ]    # Matriz de conectividade | Matriz de nós | Elementos

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
for no1,no2,no3 in elements:
    x1,x2,x3 = coords[[no1,no2,no3]]

    l = (x3 - x1)
    k11 = 7/(3*l)
    k12 = -8/(3*l)
    k13 = 1/(3*l)
    k33 = 7/(3*l)
    k22 = 16/(3*l)
    k21 = k12
    k23 = -8/(3*l)
    k32 = k23
    k31 = k13

    Ke = [
        [k11,k12,k13],
        [k21,k22,k23],
        [k31,k32,k33],
    ]
    Fe = [
        l/6,
        4*l/6,
        l/6
    ]

    #=========  Assemblagem  =========
    nos_p = [no1,no2,no3]
    for i in range(3):
        F[nos_p[i], 0] += f*Fe[i]
        for j in range(3):
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

#===============  Interpolação dos dados  ===============
x_grid = np.linspace(0,1,n_grid)
u_grid = np.zeros(n_grid)
a_grid = an(x_grid)
tol = 1e-9

for i,xg in enumerate(x_grid):
    for no1,no2,no3 in elements:
        x1,x2,x3 = coords[[no1,no2,no3]]

        if x1 <= xg <= x3 + tol:
            e = 2*(xg - x2)/(x3 - x1)
            N1 = e*(e-1)/2
            N2 = (1+e)*(1-e)
            N3 = e*(e+1)/2
            
            u_1 = ux[no1]
            u_2 = ux[no2]
            u_3 = ux[no3]

            u_val = N1*u_1 + N2*u_2 + N3*u_3

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