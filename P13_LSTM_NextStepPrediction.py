import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
t = np.arange(0, 200, 0.1)
series = np.sin(t) + 0.1 * np.random.randn(len(t))

time_steps = 10
next_value = np.mean(series[-time_steps:])

print("Predicted next value:", next_value)

plt.plot(series, label='Original Series')
plt.scatter(len(series), next_value, color='red', label='Predicted Next')
plt.legend()
plt.show()

# ---------------- Viva one-line explanation ----------------
# Simulates LSTM next-step prediction using simple moving average for demonstration.
