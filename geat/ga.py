import math
import random
import numpy as np
import matplotlib.pyplot as plt

UPPERBOUND = 1
LOWERBOUND = -0.5
GENERATIONS = 100
POPULATION_SIZE = 10

"""
Uncomment the commented code for solution to Question 1
"""

class GA():
    def __init__(self, population_size, generations, lower_bound, upper_bound):
        self.lowerbound = lower_bound
        self.upperbound = upper_bound
        self.generations = generations
        self.population_size = population_size

        self.avg_fit = []
        self.best_fit = []
        self.worst_fit = []
        self.top_individual = []
    
    def create_population(self):
        population = []
        for i in range(self.population_size):
            x_i = random.randint(self.lowerbound*10000, self.upperbound*10000)
            xs = 1 if x_i > 0 else -1
            x_i = abs(x_i)
            str_x_i = str(x_i)
            while len(str_x_i) < 4:
                str_x_i = '0' + str_x_i

            decision_variable = [int(x) for x in str_x_i]
            decision_variable.insert(0, xs)
            population.append(decision_variable)

        return population

    def convert_to_int(self, decision_variable):
        x_i = int("".join(str(i) for i in decision_variable[1:]))
        x_i /= 10000
        x_i *= decision_variable[0]
        return x_i

    def fitness_function(self, decision_variable):
        decision_variable = self.convert_to_int(decision_variable)
        return decision_variable * math.sin(10*math.pi*decision_variable) + 1

    # Roulette wheel selection
    def proportionate_selection(self, population, parents):
        selection = []
        individual_fitness = [self.fitness_function(individual) for individual in population]
        total_fitness = sum(individual_fitness)
        normalized_fitness = [i / total_fitness for i in individual_fitness]
        for i in range(parents):
            selection.append(normalized_fitness.index(np.random.choice(normalized_fitness, replace=True)))

        return selection

    # Single point crossover
    def crossover(self, parent_a, parent_b):
        N = len(parent_a)
        cross_point = random.randint(0, N)
        alleles_a = parent_a[cross_point:]
        alleles_b = parent_b[cross_point:]
        parent_a[cross_point:] = alleles_b
        parent_b[cross_point:] = alleles_a
        return parent_a, parent_b

    def mutate(self, parent_a, parent_b):
        parents = [parent_a, parent_b]
        mutated_allele = random.randint(0, len(parent_a) - 1)
        for i in range(2):
            if mutated_allele == 0:
                mutation = (-1) ** (random.random() > 0.5)
            else:
                mutation = random.randint(0, 9)
            
            parents[i][mutated_allele] = mutation

        return parent_a, parent_b

    def evolve(self, old_population, parents=2):
        next_generation = []
        mutation_probability = 0.05
        crossover_probability = 0.10

        for i in range(self.population_size // 2):
            first, second = self.proportionate_selection(old_population, parents)
            parent_a = old_population[first]
            parent_b = old_population[second]

            if np.random.random() < crossover_probability:
                parent_a, parent_b = self.crossover(parent_a, parent_b)

            if np.random.random() < mutation_probability:
                parent_a, parent_b = self.mutate(parent_a, parent_b)


            next_generation.append(parent_a)
            next_generation.append(parent_b)

        self.gather_data(old_population)
        return next_generation

    def gather_data(self, population):
        fitness = [self.fitness_function(individual) for individual in population]
        self.worst_fit.append(min(fitness))
        self.best_fit.append(max(fitness))
        self.avg_fit.append(sum(fitness) / len(fitness))
        self.top_individual.append(self.convert_to_int(population[fitness.index(max(fitness))]))

    def plot(self):
        _, axis = plt.subplots(1,3)
        axis[0].plot(self.best_fit, label='Best Fitness')
        axis[0].plot(self.worst_fit, label='Worst Fitness')
        axis[0].plot(self.avg_fit, label='Average Fitness')
        axis[0].set_xlabel('Generations')
        axis[0].set_ylabel('Fitness')
        axis[0].set_title('Fitness Evolution')
        axis[0].legend()
        axis[1].plot(self.top_individual)
        axis[1].set_xlabel('Generations')
        axis[1].set_ylabel('Decision Variable')
        axis[1].set_title('Decision Variable Evolution')
        
        x = np.linspace(-0.5, 1, 15000)
        y = x * np.sin(10*np.pi*x) + 1
        axis[2].plot(x, y)
        axis[2].set_xlabel('Value of x')
        axis[2].set_ylabel('F(x)')
        axis[2].set_title('Function f')
        plt.show()
    

if __name__ == '__main__':
    i = 1
    ga = GA(POPULATION_SIZE, GENERATIONS, LOWERBOUND, UPPERBOUND)
    population = ga.create_population()

    while True:
        print(f'\nGENERATION {i}')
        for individual in population:
            print(ga.convert_to_int(individual))
        if i == GENERATIONS:
            break
        i += 1
        
        population = ga.evolve(population)
        most_fit = sorted(population, key=ga.fitness_function)[-1]
        print('\n Final result')
        print(ga.convert_to_int(most_fit), ga.fitness_function(most_fit))

    ga.plot()

