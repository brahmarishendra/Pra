import numpy as np
import matplotlib.pyplot as plt

fan = np.arange(0, 101, 1)

def trimf(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))

slow = np.fmin(0.2, trimf(fan, 0, 0, 50))
medium = np.fmin(0.6, trimf(fan, 25, 50, 75))
fast = np.fmin(0.4, trimf(fan, 50, 100, 100))

aggregated = np.fmax(slow, np.fmax(medium, fast))

centroid = np.sum(aggregated * fan) / np.sum(aggregated)

cumulative = np.cumsum(aggregated)
bisector = fan[np.where(cumulative >= cumulative[-1] / 2)[0][0]]

max_val = np.max(aggregated)
mom = np.mean(fan[aggregated == max_val])
som = np.min(fan[aggregated == max_val])
lom = np.max(fan[aggregated == max_val])

print("Centroid:", round(centroid, 2))
print("Bisector:", bisector)
print("MOM:", mom)
print("SOM:", som)
print("LOM:", lom)

plt.plot(fan, aggregated, label="Aggregated Output")
plt.axvline(centroid, color='r', label="Centroid")
plt.axvline(bisector, color='g', label="Bisector")
plt.axvline(mom, color='b', linestyle='--', label="MOM")
plt.axvline(som, color='purple', linestyle=':', label="SOM")
plt.axvline(lom, color='orange', linestyle=':', label="LOM")
plt.xlabel("Fan Speed")
plt.ylabel("Membership")
plt.title("Defuzzification Techniques")
plt.legend()
plt.grid()
plt.show()

# ---------------- Viva one-line explanation ----------------
# Shows multiple defuzzification methods to convert aggregated fuzzy fan speed into crisp numerical output.
