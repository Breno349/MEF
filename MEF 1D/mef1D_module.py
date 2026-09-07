import numpy as np

def gen_mesh(L0,L1,n,od=2):
    coords = np.linspace(L0,L1,n)
    elements = np.array([[j for j in range(0,od)]] + [ [j+k for k in range(0,od)] for j in range(od-1,n-1,od-1) ])
    if (elements.flatten()[:] >= n).any():
        print("ERRO: Número de nós insuficiente")
        exit(1)
    return (coords,elements)

def calc(EA,f,coords,elements,n,od=2):
    K = np.zeros((n,n))
    F = np.zeros((n,1))

    for el in elements:
        nos = [k for k in el]
        x = [ coords[k] for k in nos]
        l = x[-1] - x[0]
        if od == 2:
            Ke = [
                [1/l, -1/l],
                [-1/l, 1/l]
            ]
            Fe = [
                l/2,
                l/2
            ]
        elif od == 3:
            Ke = [
                [7/(3*l), -8/(3*l), 1/(3*l)],
                [-8/(3*l), 16/(3*l), -8/(3*l)],
                [1/(3*l), -8/(3*l), 7/(3*l)]
            ]
            Fe = [
                l/6,
                4*l/6,
                l/6
            ]
        elif od == 4:
            Ke = [
                [3.7/l, -4.725/l, 1.35/l, -0.325/l],
                [-4.725/l, 10.8/l, -7.425/l, 1.35/l],
                [1.35/l, -7.425/l, 10.8/l, -4.725/l],
                [-0.325/l, 1.35/l, -4.725/l, 3.7/l],
            ]
            Fe = [
                (1/8)*l,
                (3/8)*l,
                (3/8)*l,
                (1/8)*l,
            ]
        else:
            print("ERRO: Ordem de polinómio não suportado")
            exit(1)

        # Assemblagem
        for i in range(od):
            F[nos[i], 0] += f*Fe[i]
            for j in range(od):
                K[nos[i], nos[j]] += EA*Ke[i][j]
    return (K, F)

def cond(cd_ns, cd_vs, cn_ns, cn_vs, K, F):
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
    return (K, F)

def solve(K, F):
        ux = np.linalg.solve(K, F)
        return ux

def interp(ux,coords,elements,od=2,n_grid=100,tol=1e-9):
    x_grid = np.linspace(min(coords),max(coords),n_grid)
    u_grid = np.zeros(n_grid)

    for i,xg in enumerate(x_grid):
        for el in elements:
            nos = [k for k in el]
            x = [ coords[k] for k in nos]
            if x[0] <= xg <= x[-1] + tol:

                if od == 2:
                    N1 = (x[1]-xg)/(x[1]-x[0])
                    N2 = (xg-x[0])/(x[1]-x[0])
                    u_1 = ux[nos[0]]
                    u_2 = ux[nos[1]]
                    u_val = N1*u_1 + N2*u_2
                elif od == 3:
                    e = 2*(xg - x[1])/(x[2] - x[0])
                    N1 = e*(e-1)/2
                    N2 = (1+e)*(1-e)
                    N3 = e*(e+1)/2

                    u_1 = ux[nos[0]]
                    u_2 = ux[nos[1]]
                    u_3 = ux[nos[2]]
                    u_val = N1*u_1 + N2*u_2 + N3*u_3
                elif od == 4:
                    e = 2*(xg - ((x[3]+x[0])/2))/(x[3]-x[0])

                    N1 = (-9/16)*(e+1/3)*(e-1/3)*(e-1)
                    N2 = (27/16)*(e+1)*(e-1/3)*(e-1)
                    N3 = (-27/16)*(e+1)*(e+1/3)*(e-1)
                    N4 = (9/16)*(e+1)*(e+1/3)*(e-1/3)

                    u_1 = ux[nos[0]]
                    u_2 = ux[nos[1]]
                    u_3 = ux[nos[2]]
                    u_4 = ux[nos[3]]
                    u_val = N1*u_1 + N2*u_2 + N3*u_3 + N4*u_4
                u_grid[i] = u_val
                break
    return (x_grid, u_grid)