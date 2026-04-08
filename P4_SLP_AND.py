import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([0, 0, 0, 1])

w = np.zeros(2)
b = 0
lr = 0.1
epochs = 20

def step(x):
    return 1 if x >= 0 else 0

for _ in range(epochs):
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = step(z)
        error = y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error

print("Weights:", w)
print("Bias:", b)
print("Predictions:")
for i in range(len(X)):
    print(X[i], "->", step(np.dot(X[i], w) + b))

plt.scatter(X[:,0], X[:,1], c=y, s=100, cmap='bwr')
x_vals = np.array([-0.5, 1.5])
y_vals = -(w[0]*x_vals + b) / w[1]
plt.plot(x_vals, y_vals, '--k')
plt.xlabel("Input X1")
plt.ylabel("Input X2")
plt.title("Single-Layer Perceptron Decision Boundary")
plt.grid()
plt.show()

# ---------------- Viva one-line explanation ----------------
# Trains a perceptron to learn AND logic and visualizes its decision boundary.
