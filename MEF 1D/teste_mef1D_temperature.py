import numpy as np

L = 0.5
n = 101
coords = np.linspace(0,L,n)
elements = [(j-1,j) for j in range(1,n)]

p = 2700.0
c = 900.0
k = 205.0
q = 0
dt = 2
TB = 40
T0 = 5

K_g = np.zeros((n,n))
M_g = np.zeros((n,n))
F_g = np.zeros((n,1))

for no1,no2 in elements:
    x1,x2 = coords[[no1,no2]]
    he = x2 - x1
    Me = (p*c*he/6)* np.array([
        [2,1],
        [1,2]
    ])
    Ke = (k/he) * np.array([
        [1,-1],
        [-1,1]
    ])
    Fe = (q*he/2) * np.array([
        1,
        1
    ])
    nos_p = [no1,no2]
    for i in range(2):
        F_g[nos_p[i], 0] += Fe[i]
        for j in range(2):
            K_g[nos_p[i], nos_p[j]] += Ke[i][j]
            M_g[nos_p[i], nos_p[j]] += Me[i][j]

total = 600
u_curr = np.ones((n,1))*TB
u_global = []
for t in range(0,total,dt):
    u_global.append(u_curr)
    
    K_efc = (1/dt) * M_g + K_g
    F_efc = F_g + (1/dt) * (M_g @ u_curr)

    # aplicando condições de contorno
    no = 0
    F_efc[:, 0] -= K_efc[:, no]*T0
    K_efc[no, :] = 0
    K_efc[:, no] = 0
    K_efc[no, no] = 1
    F_efc[no, 0] = T0

    u_next = np.linalg.solve(K_efc, F_efc)
    u_curr = u_next


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# --- CONFIGURAÇÃO DO MAPA DE CORES (COLORMAP) ---
cmap = cm.jet               # Escolha 'hot' ou 'jet'
#norm = Normalize(vmin=T0, vmax=TB) # Normaliza os valores entre a menor e a maior temperatura
norm = Normalize(vmin=T0-10, vmax=TB+10) 

# --- CONFIGURAÇÃO DA FIGURA ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
ax.set_ylim(T0 - 5, TB + 5)
ax.set_xlabel("Posição na Barra (m)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Evolução Térmica com Gradiente de Temperatura (MEF)")
ax.grid(True)

# 1. Preparar a estrutura de segmentos de reta para o LineCollection
# Vincula cada ponto x ao seu vizinho criando pequenos pedaços de linha
points = np.array([coords, np.zeros_like(coords)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 2. Criar a coleção de linhas usando o colormap escolhido
lc = LineCollection(segments, cmap=cmap, norm=norm, lw=3)
ax.add_collection(lc)

# 3. Adicionar uma barra de cores (Colorbar) lateral para referência
cbar = fig.colorbar(lc, ax=ax, label="Temperatura (°C)")

# Cronômetro digital interno
time_text = ax.text(0.75, 0.90, "", transform=ax.transAxes, fontsize=14, 
                    fontweight="bold", bbox=dict(facecolor='white', alpha=0.6))

# --- FUNÇÃO DE ATUALIZAÇÃO DA ANIMAÇÃO ---
def atualizar(i):
    temperaturas = u_global[i].flatten()
    
    # 1. Atualiza os segmentos espaciais
    new_points = np.array([coords, temperaturas]).T.reshape(-1, 1, 2)
    new_segments = np.concatenate([new_points[:-1], new_points[1:]], axis=1)
    lc.set_segments(new_segments)
    
    temperaturas_medias = (temperaturas[:-1] + temperaturas[1:]) / 2
    lc.set_array(temperaturas_medias)
    
    # --- A MÁGICA DO AUTOSCALE DE CORES ---
    # Encontra o menor e o maior valor EXATOS do frame atual
    t_min = temperaturas.min()
    t_max = temperaturas.max()
    
    # Se a barra começar a esfriar muito e os valores mudarem,
    # o colormap se reajusta sozinho para nunca estourar no branco!
    # Adicionamos uma folga dinâmica de 5 graus no topo apenas se t_max for alto
    lc.set_clim(vmin=t_min-10, vmax=t_max + 10) 
    # --------------------------------------
    
    # Atualiza o cronômetro
    tempo_simulado = i * dt
    time_text.set_text(f"Tempo: {tempo_simulado}s")
    
    return lc, time_text


# Roda a animação fluida

ani = animation.FuncAnimation(fig, atualizar, frames=len(u_global), interval=50, blit=True)

# Definindo o escritor de vídeo (ajusta os frames por segundo - FPS)
writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)

print("Salvando animação em MP4... Por favor, aguarde.")
ani.save("simulacao_mef_calor.mp4", writer=writer)
print("Vídeo salvo com sucesso como 'simulacao_mef_calor.mp4'!")

plt.show()