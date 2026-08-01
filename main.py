import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Build network
# Node 0 = center
# Nodes 1-8 = inner ring
# Nodes 9-16 = outer ring
# -------------------------------------------------

N = 17
A = np.zeros((N, N))

def connect(i, j):
    A[i, j] = 1
    A[j, i] = 1

# Center connected to all inner nodes
for i in range(1, 9):
    connect(0, i)

# Inner ring
for i in range(1, 9):
    nxt = 1 + (i % 8)
    connect(i, nxt)

# Outer ring
for i in range(9, 17):
    nxt = 9 + ((i - 9 + 1) % 8)
    connect(i, nxt)

# Each inner node connects to its corresponding outer node
for i in range(8):
    connect(1+i, 9+i)

# -------------------------------------------------
# Diffusion simulation
# -------------------------------------------------

degree = np.diag(A.sum(axis=1))
L = degree - A

dt = 0.05
steps = 400

# Initial energy
E = np.zeros(N)
E[0] = 170.0       # all energy initially in center

history = [E.copy()]

for _ in range(steps):
    E = E - dt * (L @ E)
    history.append(E.copy())

history = np.array(history)

print("Final energies:")
print(np.round(history[-1],3))
print("Mean energy =", history[-1].mean())

# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(history[:,0], lw=3, label="Center")

for i in range(1,9):
    plt.plot(history[:,i], color='tab:green', alpha=0.5)

for i in range(9,17):
    plt.plot(history[:,i], color='tab:orange', alpha=0.5)

plt.axhline(history[-1].mean(),
            color='k',
            ls='--',
            label='Equilibrium')

plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Energy diffusion on a 17-node network")
plt.legend()
plt.grid(True)
plt.show()
