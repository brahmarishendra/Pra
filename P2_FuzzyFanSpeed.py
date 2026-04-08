import numpy as np
import matplotlib.pyplot as plt

temp = np.arange(0, 41, 1)
fan = np.arange(0, 101, 1)

def trimf(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))

temp_low = trimf(temp, 0, 0, 20)
temp_med = trimf(temp, 10, 20, 30)
temp_high = trimf(temp, 20, 40, 40)

fan_slow = trimf(fan, 0, 0, 50)
fan_med = trimf(fan, 25, 50, 75)
fan_fast = trimf(fan, 50, 100, 100)

plt.plot(temp, temp_low, label="Low")
plt.plot(temp, temp_med, label="Medium")
plt.plot(temp, temp_high, label="High")
plt.title("Temperature Membership Functions")
plt.xlabel("Temperature")
plt.ylabel("Membership")
plt.legend()
plt.grid()
plt.show()

input_temp = 28
low_deg = np.interp(input_temp, temp, temp_low)
med_deg = np.interp(input_temp, temp, temp_med)
high_deg = np.interp(input_temp, temp, temp_high)

rule1 = np.fmin(low_deg, fan_slow)
rule2 = np.fmin(med_deg, fan_med)
rule3 = np.fmin(high_deg, fan_fast)

aggregated = np.fmax(rule1, np.fmax(rule2, rule3))

fan_speed = np.sum(aggregated * fan) / np.sum(aggregated)

print("Input Temperature:", input_temp)
print("Fan Speed Output:", round(fan_speed, 2))

plt.plot(fan, fan_slow, '--', alpha=0.5)
plt.plot(fan, fan_med, '--', alpha=0.5)
plt.plot(fan, fan_fast, '--', alpha=0.5)
plt.fill_between(fan, aggregated, alpha=0.6)
plt.axvline(fan_speed, color='r', label='Defuzzified Output')
plt.title("Fuzzy Output (Fan Speed)")
plt.xlabel("Fan Speed")
plt.ylabel("Membership")
plt.legend()
plt.grid()
plt.show()

# ---------------- Viva one-line explanation ----------------
# Mamdani Fuzzy Logic Controller calculates fan speed by fuzzifying temperature, applying rules, aggregating outputs, and defuzzifying.
