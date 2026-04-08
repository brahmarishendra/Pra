import numpy as np
import matplotlib.pyplot as plt

temperature = np.arange(0, 41, 1)
fan_speed = np.arange(0, 101, 1)

def trimf(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))

temp_low = trimf(temperature, 0, 0, 20)
temp_med = trimf(temperature, 10, 20, 30)
temp_high = trimf(temperature, 20, 40, 40)

fan_slow = trimf(fan_speed, 0, 0, 50)
fan_mod = trimf(fan_speed, 25, 50, 75)
fan_fast = trimf(fan_speed, 50, 100, 100)

input_temp = 28
low_deg = np.interp(input_temp, temperature, temp_low)
med_deg = np.interp(input_temp, temperature, temp_med)
high_deg = np.interp(input_temp, temperature, temp_high)

rule1 = np.fmin(low_deg, fan_slow)
rule2 = np.fmin(med_deg, fan_mod)
rule3 = np.fmin(high_deg, fan_fast)

aggregated = np.fmax(rule1, np.fmax(rule2, rule3))

fan_out = np.sum(aggregated * fan_speed) / np.sum(aggregated)

print("Input Temperature:", input_temp)
print("Fan Speed Output:", round(fan_out, 2))

plt.plot(fan_speed, fan_slow, '--', alpha=0.5)
plt.plot(fan_speed, fan_mod, '--', alpha=0.5)
plt.plot(fan_speed, fan_fast, '--', alpha=0.5)
plt.fill_between(fan_speed, aggregated, alpha=0.6)
plt.axvline(fan_out, color='r', label='Defuzzified Output')
plt.title("Fuzzy Fan Controller Output")
plt.xlabel("Fan Speed")
plt.ylabel("Membership")
plt.legend()
plt.grid()
plt.show()

# ---------------- Viva one-line explanation ----------------
# Computes fan speed from temperature using manual fuzzy logic and defuzzification.
