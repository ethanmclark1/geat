import math
import random
import numpy as np
import matplotlib.pyplot as plt

UPPERBOUND = 1
LOWERBOUND = -0.5
GENERATIONS = 300
POPULATION_SIZE = 10

"""
    Objective 1 = sin(3pix)
    Objective 2 = sin(5pix)
"""

class MOGA():
    def __init__(self, population_size, generations, lower_bound, upper_bound):
        self.lowerbound = lower_bound
        self.upperbound = upper_bound
        self.generations = generations
        self.population_size = population_size

        self.best_fit = []
        self.best_pareto_fit = []
        self.top_individual = []
        self.top_pareto_individual = []

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

    # Convert decision variable to int
    def convert(self, decision_variable):
        x_i = int("".join(str(i) for i in decision_variable[1:]))
        x_i /= 10000
        x_i *= decision_variable[0]
        return x_i

    def sin_fit(self, decision_variable):
        decision_variable = self.convert(decision_variable)
        return np.sin(3*np.pi*decision_variable)
    
    def cos_fit(self, decision_variable):
        decision_variable = self.convert(decision_variable)
        return -np.cos(3*np.pi*decision_variable)

    def sin_fit1(self, decision_variable):
        return np.sin(3*np.pi*decision_variable)

    def cos_fit1(self, decision_variable):
        return -np.cos(5*np.pi*decision_variable)

    def rank_select(self, pop, parents=2):
        probs = []
        Z = 0.05
        U = ((Z*(POPULATION_SIZE-1)) / 2) + (1/POPULATION_SIZE)
        probs = [U - i*Z for i in range(POPULATION_SIZE)]

        population_fitness = []
        for individual in pop:
            fitness = self.sin_fit(individual) + self.cos_fit(individual)
            population_fitness.append([fitness, individual])
        ranked = sorted(population_fitness, reverse=True)
        selection = random.choices(ranked, weights=probs, k=parents)
        return selection[0][1], selection[1][1]

    def pareto_rank(self, pareto_pop, parents=2):
        Z = 0.05
        U = ((Z*(POPULATION_SIZE-1)) / 2) + (1/POPULATION_SIZE)
        probs = [U - i*Z for i in range(POPULATION_SIZE)]

        domination = []
        non_dominated_count = 0
        for x in pareto_pop:
            dominated = 0
            for y in pareto_pop:
                if x != y and \
                    self.sin_fit(x) <= self.sin_fit(y) and self.cos_fit(x) <= self.cos_fit(y):
                    dominated += 1
            domination.append([dominated, x])
            if not dominated: non_dominated_count += 1

        domination = sorted(domination)
        for i in range(non_dominated_count):
            probs[i] = max(probs)
        selection = random.choices(domination, weights=probs, k=parents)
        return selection[0][1], selection[1][1]
                
    def pareto_frontier(self):
        dominated = 0
        pareto_front = []
        for i in range(-500, 1000, 1):
            dominated = 0
            x_val = i / 1000
            for j in range(-500, 1000, 1):
                y_val = j / 1000
                if x_val != y_val and \
                    self.sin_fit1(x_val) <= self.sin_fit1(y_val) and self.cos_fit1(x_val) <= self.cos_fit1(y_val):
                    dominated += 1
            if not dominated: pareto_front.append(x_val)

        return pareto_front

    def crossover(self, parent_a, parent_b,):
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

    def evolve(self, pop, pareto_pop):
        next_gen = []
        next_pareto_gen = []
        mutation_prob = 0.20
        crossover_prob = 0.30

        self.gather_data(pop, pareto_pop)
        for i in range(self.population_size // 2):
            child_a, child_b = self.rank_select(pop)
            pareto_child_a, pareto_child_b = self.pareto_rank(pareto_pop)

            if random.random() < crossover_prob:
                child_a, child_b = self.crossover(child_a, child_b)
                pareto_child_a, pareto_child_b = self.crossover(pareto_child_a, pareto_child_b)
            if random.random() < mutation_prob:
                child_a, child_b = self.mutate(child_a, child_b)
                pareto_child_a, pareto_child_b = self.mutate(pareto_child_a, pareto_child_b)
            
            next_gen.extend([child_a, child_b])
            next_pareto_gen.extend([pareto_child_a, pareto_child_b])
        
        return next_pareto_gen

    def gather_data(self, next_gen, next_pareto_gen):
        fitness = [self.sin_fit(individual) + self.cos_fit(individual) for individual in next_gen]
        self.best_fit.append(max(fitness))
        self.top_individual.append(self.convert(next_gen[fitness.index(max(fitness))]))

        pareto_fitness = [self.sin_fit(individual) + self.cos_fit(individual) for individual in next_pareto_gen]
        self.best_pareto_fit.append(max(pareto_fitness))
        self.top_pareto_individual.append(self.convert(next_pareto_gen[pareto_fitness.index(max(pareto_fitness))]))


    def plot(self):
        _, axis = plt.subplots(1,2)
        pareto_front = self.pareto_frontier()
        pf_sin = [self.sin_fit1(x) for x in pareto_front]
        pf_cos = [self.cos_fit1(x) for x in pareto_front]
        pf = [[sin_fit, cos_fit] for sin_fit, cos_fit in zip(pf_sin, pf_cos)]
        np_top = np.asarray(self.top_individual)
        sin = np_top * np.sin(10*np.pi*np_top)
        cos = 2.5 * np_top * np.cos(3*np.pi*np_top)
        axis[0].scatter(sin, cos, label='Decision Variable')
        axis[0].scatter(pf_sin, pf_cos, label='Non-Dominated Frontier')
        axis[0].set_xlabel('Sin Objective')
        axis[0].set_ylabel('Cos Objective')
        axis[0].set_title('Simple-Ranking Evolution')
        axis[0].legend()

        np_top = np.asarray(self.top_pareto_individual)
        sin = np_top * np.sin(10*np.pi*np_top)
        cos = 2.5 * np_top * np.cos(3*np.pi*np_top)
        axis[1].scatter(sin, cos, label='Decision Variable')
        axis[1].scatter(pf_sin, pf_cos, label='Non-Dominated Frontier')
        axis[1].set_xlabel('Sin Objective')
        axis[1].set_ylabel('Cos Objective')
        axis[1].set_title('Pareto-Ranking Evolution')
        axis[1].legend()
        plt.show()

if __name__ == '__main__':        
    i = 0
    moga = MOGA(POPULATION_SIZE, GENERATIONS, LOWERBOUND, UPPERBOUND)
    pop = moga.create_population()
    pareto_pop = moga.create_population()

    while True:
        print(f'\nGENERATION {i}')
        for individual, p_individual in zip(pop, pareto_pop):
            print(f'Decision Var: {moga.convert(individual)}\tPareto Decision Var: {moga.convert(p_individual)}')
        if i == GENERATIONS :
            break
        i += 1

        pareto_pop = moga.evolve(pop, pareto_pop)

    moga.plot()      

