import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

n = 4
domain = np.linspace(0,1,n);
h = domain[1]-domain[0];
elements = [[j-1,j] for j in range(1,n)]

"""
    N1 = (x2 - x)/(x2 - x1);
    N2 = (x - x1)/(x2 - x1);
    dN1 = -1/h;
    dN2 = 1/h;
"""

K = np.zeros((n,n))
F = np.zeros((n,1))

for e in elements:
    nos = domain[e]
    N1 = lambda x: (nos[1] - x)/(h);
    N2 = lambda x: (x - nos[0])/(h);
    dN1 = -1/h;
    dN2 = 1/h;
    k_ij = [
        [dN1*dN1, dN1*dN2],
        [dN2*dN1, dN2*dN2]
    ];
    f_i = [
        lambda x: 2*N1(x),
        lambda x: 2*N2(x)
    ];
    for i,col in enumerate(k_ij):
        f_i[i] = integrate.quad(f_i[i], nos[0], nos[1])[0]
        for j,row in enumerate(col):
            k_ij[i][j] = integrate.quad(lambda x: row, nos[0], nos[1])[0]
    n1 = e[0]
    n2 = e[1]
    K[n1,n1] += k_ij[0][0]
    K[n1,n2] += k_ij[0][1]
    K[n2,n1] += k_ij[1][0]
    K[n2,n2] += k_ij[1][1]
    F[n1,0] += f_i[0]
    F[n2,0] += f_i[1]

#print(K)
#print(F)

# condições de contorno de dirichlet
# u(0) = 1
F[0, 0] = 1
K[0, :] = 0
K[0, 0] = 1
for i in range(1,n):
    F[i, 0] -= K[i, 0]*1
    K[i, 0] = 0

u = np.linalg.solve(K,F)

x = np.linspace(0,1,100)
v = -x**2 + 2*x + 1;

# Plotar resultados
plt.plot(x, v)
plt.plot(domain, u, '--')
plt.grid()
plt.legend(["Analitica", "Aproximação"])
plt.show()