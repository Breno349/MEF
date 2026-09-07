import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from read_mesh import read_mesh

#=========  Importando domínio  =========
elements, coords, groups = read_mesh('malha.su2')

# Função para procurar aresta
def find_edge(grupo,ns):
    i = -1
    for no1,no2 in grupo:
        i += 1
        if no1 in ns:
            if no2 in ns:
                return ns.index(grupo[i][0]),ns.index(grupo[i][1])
    return None

# Número de elementos e nós
num_ell = len(elements)
num_nos = len(coords)

# Fatores constantes
ax = 1
ay = 1
B = 0
g = 0

# Condições de contorno de diriclhet [no1,no2,...]
nos_dirc = [
    np.unique(np.array(groups['ladoe']).flatten(), axis=0),
    np.unique(np.array(groups['ladod']).flatten(), axis=0),
    np.unique(np.array(groups['alto']).flatten(), axis=0),
    np.unique(np.array(groups['baixo']).flatten(), axis=0),
]
u_dirc = [0, 0, 0, 1]
# Condições de contorno de newmann [(no1,no2),(no4,no5)..]
nos_newm = []
q_newm = []

#=========  Alocando as matrizes =========
K = lil_matrix((num_nos, num_nos))
F = np.zeros(num_nos)

#=========  Fazendo cálculo de cada matriz elementar e realizando assemblagem  =========
for no1,no2,no3 in elements:
    x1,y1,x2,y2,x3,y3 = coords[[no1,no2,no3]].flatten()
    Ae = (x2-x1)*(y3-y1)/2 - (x3-x1)*(y2-y1)/2
    J = Ae*2

    b = [y2 - y3, y3 - y1, y1 - y2]
    c = [x3 - x2, x1 - x3, x2 - x1]

    Me = np.zeros((3,3))
    Te = np.zeros((3,3))
    Fe = np.zeros((3,1))
    Pe = np.zeros((3,1))

    for i in range(3):
        Fe[i, 0] = g*Ae/3
        for j in range(3):
            Me[i,j] = -(ax*b[i]*b[j] + ay*c[i]*c[j])/(4*Ae)
            if i == j:
                Te[i, j] = B*Ae/6
            else:
                Te[i, j] = B*Ae/12

    nos_p = [no1,no2,no3]

    #=========  Condições de contorno de Newmann  =========
    for k,nos_nw in enumerate(nos_newm):
        edge = find_edge(nos_nw, nos_p)
        q = q_newm[k]
        if edge is not None:
            i_a, i_b = edge                       # posições LOCAIS (0,1,2) na aresta
            no_a, no_b = nos_p[i_a], nos_p[i_b]   # nós GLOBAIS correspondentes
            xa, ya = coords[no_a]
            xb, yb = coords[no_b]
            l = np.sqrt((xb-xa)**2 + (yb-ya)**2)
            Pe[i_a] += -q*l/2
            Pe[i_b] += -q*l/2

    #=========  Assemblagem  =========
    for i in range(3):
        F[nos_p[i]] += Fe[i,0] + Pe[i,0]
        for j in range(3):
            K[nos_p[i], nos_p[j]] += Me[i, j] + Te[i, j]

# Função de aplicar condição de contorno de diriclhet
def dirc(no, val):
    global F
    F -= K[:, no].toarray().flatten() * val
    K[no, :] = 0
    K[:, no] = 0
    K[no, no] = 1
    F[no] = val

#=========  Condições de contorno de Diriclhet  =========
for i,nos in enumerate(nos_dirc):
    for no in nos:
        dirc(no, u_dirc[i])

#=========  Resolver o sistema  =========
K = K.tocsr()
ux = spsolve(K, F)

# Erro.. Grid..

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessário para projection='3d'
x = coords[:, 0]
y = coords[:, 1]
z = ux
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(x, y, z, triangles=elements, cmap='jet', 
                        linewidth=0, antialiased=True)
fig.colorbar(surf, ax=ax, shrink=0.6, label='u')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('Solução MEF 2D — superfície')
plt.tight_layout()
plt.savefig('resultado_3d.png', dpi=200)
plt.show()