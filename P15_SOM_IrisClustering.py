import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

m, n = 7, 7
input_len = X.shape[1]

weights = np.random.random((m, n, input_len))

epochs = 1000
initial_lr = 0.6
initial_radius = max(m, n) / 2

for epoch in range(epochs):
    lr = initial_lr * np.exp(-epoch / epochs)
    radius = initial_radius * np.exp(-epoch / epochs)

    for sample in X:
        dist = np.linalg.norm(weights - sample, axis=2)
        bmu_x, bmu_y = np.unravel_index(np.argmin(dist), dist.shape)

        for i in range(m):
            for j in range(n):
                d = np.sqrt((i - bmu_x)**2 + (j - bmu_y)**2)
                if d <= radius:
                    influence = np.exp(-(d**2) / (2 * (radius**2)))
                    weights[i, j] += lr * influence * (sample - weights[i, j])

plt.figure(figsize=(7,7))
markers = ['o', 's', '^']
colors = ['red', 'green', 'blue']

for i, sample in enumerate(X):
    dist = np.linalg.norm(weights - sample, axis=2)
    bmu_x, bmu_y = np.unravel_index(np.argmin(dist), dist.shape)
    plt.scatter(bmu_x, bmu_y,
                c=colors[y[i]],
                marker=markers[y[i]],
                s=60)

plt.title("Self-Organizing Map (Proper Clustering)")
plt.grid()
plt.show()

# ---------------- Viva one-line explanation ----------------
# Demonstrates clustering of the Iris dataset using a Self-Organizing Map.
