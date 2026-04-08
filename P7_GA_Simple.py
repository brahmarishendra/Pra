import random

def fitness(x):
    return x * x

def initialize_population(size):
    return [random.uniform(-10, 10) for _ in range(size)]

def selection(population):
    a, b = random.sample(population, 2)
    return a if fitness(a) < fitness(b) else b

def crossover(p1, p2):
    alpha = random.random()
    return alpha * p1 + (1 - alpha) * p2

def mutation(child, rate=0.1):
    if random.random() < rate:
        child += random.uniform(-1, 1)
    return child

def genetic_algorithm():
    population_size = 20
    generations = 50

    population = initialize_population(population_size)

    for _ in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1 = selection(population)
            parent2 = selection(population)
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)
        population = new_population

    best = min(population, key=fitness)
    print("Best solution:", best)
    print("Minimum value:", fitness(best))

genetic_algorithm()

# ---------------- Viva one-line explanation ----------------
# Uses selection, crossover, and mutation to find x that minimizes x^2.
