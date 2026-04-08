import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
t = np.arange(0, 100, 0.1)
data = np.sin(t) + 0.1 * np.random.randn(len(t))

time_steps = 10
predicted_value = np.mean(data[-time_steps:])

print("Predicted next value:", predicted_value)

plt.plot(data, label='Original Series')
plt.scatter(len(data), predicted_value, color='red', label='Predicted Next')
plt.legend()
plt.show()

# ---------------- Viva one-line explanation ----------------
# Simulates LSTM next-step prediction with moving average to demonstrate prediction logic.
