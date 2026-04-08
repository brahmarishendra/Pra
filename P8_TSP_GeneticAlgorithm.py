import random
import numpy as np

dist_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

num_cities = len(dist_matrix)

def fitness(tour):
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1], tour[0]]

def initialize_population(size):
    population = []
    for _ in range(size):
        tour = list(range(num_cities))
        random.shuffle(tour)
        population.append(tour)
    return population

def selection(pop):
    a, b = random.sample(pop, 2)
    return a if fitness(a) < fitness(b) else b

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(num_cities), 2))
    child = [-1]*num_cities
    child[start:end+1] = parent1[start:end+1]
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city
    return child

def mutate(tour, rate=0.2):
    if random.random() < rate:
        i, j = random.sample(range(num_cities), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def ga_tsp():
    population = initialize_population(10)
    generations = 50

    for _ in range(generations):
        new_pop = []
        for _ in population:
            p1, p2 = selection(population), selection(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    best_tour = min(population, key=fitness)
    print("Best tour:", best_tour)
    print("Minimum distance:", fitness(best_tour))

ga_tsp()

# ---------------- Viva one-line explanation ----------------
# Uses GA to evolve city tours, selecting, crossing, and mutating to minimize total travel distance.
