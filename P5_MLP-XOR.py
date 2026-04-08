import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(1)
w1 = np.random.rand(2, 2)
b1 = np.random.rand(1, 2)
w2 = np.random.rand(2, 1)
b2 = np.random.rand(1, 1)

lr = 0.5
epochs = 5000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

losses = []

for _ in range(epochs):
    hidden_input = np.dot(X, w1) + b1
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, w2) + b2
    output = sigmoid(output_input)

    error = y - output
    losses.append(np.mean(error**2))

    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(w2.T) * sigmoid_derivative(hidden_output)

    w2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    w1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

print("Predictions after training:")
print(np.round(output))

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Error using Backpropagation")
plt.grid()
plt.show()

# ---------------- Viva one-line explanation ----------------
# Trains a multi-layer perceptron to solve XOR logic using backpropagation and plots training loss.
