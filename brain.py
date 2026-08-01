import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------
# Build graph
# ---------------------------------------------------

G = nx.Graph()

# Node numbering
ROOT = 0
HUB1 = 1
HUB2 = 2

G.add_nodes_from([ROOT, HUB1, HUB2])

G.add_edge(ROOT, HUB1)
G.add_edge(ROOT, HUB2)

inner1 = list(range(3,11))
inner2 = list(range(11,19))

outer1 = list(range(19,27))
outer2 = list(range(27,35))

# connect hubs to inner rings
for n in inner1:
    G.add_edge(HUB1,n)

for n in inner2:
    G.add_edge(HUB2,n)

# connect inner rings
for ring in [inner1,inner2]:
    for i in range(8):
        G.add_edge(ring[i], ring[(i+1)%8])

# connect outer rings
for ring in [outer1,outer2]:
    for i in range(8):
        G.add_edge(ring[i], ring[(i+1)%8])

# radial connections
for i in range(8):
    G.add_edge(inner1[i], outer1[i])
    G.add_edge(inner2[i], outer2[i])

# ---------------------------------------------------
# Layout
# ---------------------------------------------------

pos = {}

pos[ROOT]=(0,4)

pos[HUB1]=(-3,2)
pos[HUB2]=(3,2)

theta=np.linspace(0,2*np.pi,8,endpoint=False)

r1=1
r2=2

# left cluster
for i,a in enumerate(theta):
    pos[inner1[i]]=(-3+r1*np.cos(a),2+r1*np.sin(a))
    pos[outer1[i]]=(-3+r2*np.cos(a),2+r2*np.sin(a))

# right cluster
for i,a in enumerate(theta):
    pos[inner2[i]]=(3+r1*np.cos(a),2+r1*np.sin(a))
    pos[outer2[i]]=(3+r2*np.cos(a),2+r2*np.sin(a))

colors=[]

for n in G.nodes():

    if n==ROOT:
        colors.append("red")

    elif n in [HUB1,HUB2]:
        colors.append("dodgerblue")

    elif n in inner1 or n in inner2:
        colors.append("limegreen")

    else:
        colors.append("orange")

plt.figure(figsize=(10,8))

nx.draw_networkx(
    G,
    pos,
    node_color=colors,
    node_size=450,
    with_labels=True,
    edgecolors='black'
)

plt.title("Hierarchical Network")
plt.axis("equal")
plt.show()

# ---------------------------------------------------
# Diffusion
# ---------------------------------------------------

A = nx.to_numpy_array(G)

D = np.diag(A.sum(axis=1))
L = D - A

N = len(G)

E = np.zeros(N)
E[ROOT]=350

dt=0.05
steps=500

history=[E.copy()]

for _ in range(steps):
    E = E - dt*(L@E)
    history.append(E.copy())

history=np.array(history)

# ---------------------------------------------------
# Plot energy evolution
# ---------------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(history[:,ROOT],lw=3,label="Root")

plt.plot(history[:,HUB1],lw=2,label="Hub A")
plt.plot(history[:,HUB2],lw=2,label="Hub B")

for n in inner1+inner2:
    plt.plot(history[:,n],color="green",alpha=.2)

for n in outer1+outer2:
    plt.plot(history[:,n],color="orange",alpha=.2)

plt.axhline(history[-1].mean(),ls="--",color="k")

plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Energy Diffusion")
plt.grid(True)
plt.legend()
plt.show()

print("Final energy at each node:")
print(np.round(history[-1],3))
print("Average =",history[-1].mean())