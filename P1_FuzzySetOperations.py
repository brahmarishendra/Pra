import numpy as np

A = np.array([0.2, 0.5, 0.7, 0.9])
B = np.array([0.6, 0.3, 0.8, 0.4])

union = np.maximum(A, B)
intersection = np.minimum(A, B)
complement_A = 1 - A
complement_B = 1 - B
complement_A_and_B = 1 - np.minimum(A, B)
complement_A_or_B = 1 - np.maximum(A, B)

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)
print("Union (A ∪ B):", union)
print("Intersection (A ∩ B):", intersection)
print("Complement ¬A:", complement_A)
print("Complement ¬B:", complement_B)
print("Complement ¬(A ∩ B):", complement_A_and_B)
print("Complement ¬(A ∪ B):", complement_A_or_B)

# ---------------- Viva one-line explanation ----------------
# Demonstrates union, intersection, and complement operations on fuzzy sets.
