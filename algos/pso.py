import math
import random
import numpy as np
import matplotlib.pyplot as plt

UPPERBOUND = 1
LOWERBOUND = 0
GENERATIONS = 100
POPULATION = PARTICLES = 10
INERTIA = 0.2
GA_BEST_VAR = []
GA_BEST_FIT = []
PSO_BEST_VAR = []
PSO_BEST_FIT = []

def fit(x):
    return 1 - np.sin(5*np.pi*x)**6 * np.exp(-2*np.log(2)*((x - 0.1)/0.8)**2)

def data(ga_min_var, ga_min_fit, pso_min_var, pso_min_fit):
    GA_BEST_VAR.append(ga_min_var)
    GA_BEST_FIT.append(ga_min_fit)
    PSO_BEST_VAR.append(pso_min_var)
    PSO_BEST_FIT.append(pso_min_fit)

def plot():
    _, axis = plt.subplots(2,2)

    axis[0,0].boxplot(GA_BEST_VAR, vert=False)
    axis[0,0].plot([0.1]*3, range(3), label='Minima')
    axis[0,0].set_xlabel('Decision Variable')
    axis[0,0].set_title('Genetic Algorithm')
    axis[0,0].legend()
    axis[0,1].boxplot(PSO_BEST_VAR, vert=False)
    axis[0,1].plot([0.1]*3, range(3), label='Minima')
    axis[0,1].set_xlabel('Decision Variable')
    axis[0,1].set_title('Particle Swarm Optimization')
    axis[0,1].legend()
    axis[1,0].boxplot(GA_BEST_FIT)
    axis[1,0].plot(range(3), [0]*3, label='Minima')
    axis[1,0].set_ylabel('Fitness')
    axis[1,0].set_title('Genetic Algorithm')
    axis[1,0].legend()
    axis[1,1].boxplot(PSO_BEST_FIT)
    axis[1,1].plot(range(3), [0]*3, label='Minima')
    axis[1,1].set_ylabel('Fitness')
    axis[1,1].set_title('Particle Swarm Optimization')
    axis[1,1].legend()
    plt.show()
    
class GA():
    def __init__(self):
        self.lowerbound = LOWERBOUND
        self.upperbound = UPPERBOUND
        self.population_size = POPULATION
        self.min_var = math.inf
        self.min_fit = math.inf

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

    def rank_select(self, pop, parents=2):
        probs = []
        Z = 0.05
        U = ((Z*(POPULATION-1)) / 2) + (1/POPULATION)
        probs = [U - i*Z for i in range(POPULATION)]

        population_fitness = []
        for individual in pop:
            fitness = fit(self.convert(individual))
            population_fitness.append([fitness, individual])
        ranked = sorted(population_fitness)
        selection = random.choices(ranked, weights=probs, k=parents)
        return selection[0][1], selection[1][1]

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
        parent_a[0] = abs(parent_a[0])
        parent_b[0] = abs(parent_b[0])
        return parent_a, parent_b

    def evolve(self, pop):
        next_gen = []
        mutation_prob = 0.20
        crossover_prob = 0.30

        self.gather_data(pop)
        for i in range(self.population_size // 2):
            child_a, child_b = self.rank_select(pop)

            if random.random() < crossover_prob:
                child_a, child_b = self.crossover(child_a, child_b)
            if random.random() < mutation_prob:
                child_a, child_b = self.mutate(child_a, child_b)
            
            next_gen.extend([child_a, child_b])
        return next_gen

    def gather_data(self, next_gen):
        fitness = [fit(self.convert(individual)) for individual in next_gen]
        min_fit = min(fitness)
        min_var = self.convert(next_gen[fitness.index(min_fit)])
        if min_fit <= self.min_fit:
            self.min_var = min_var
            self.min_fit = min_fit

class PSO():
    def __init__(self):
        self.a = INERTIA
        self.lowerbound = LOWERBOUND
        self.upperbound = UPPERBOUND
        self.num_particles = PARTICLES
        self.pbest, self.gbest = None, None

    def generate(self):
        particles = []
        velocity = []
        for i in range(self.num_particles):
            x = np.random.uniform()
            v = np.random.uniform(low=-1, high=1)
            particles.append(x)
            velocity.append(v)
        
        particles = np.array(particles)
        velocity = np.array(velocity)
        self.pbest = particles
        self.gbest = self.pbest[fit(particles).argmin()]
        return particles, velocity

    def iterate(self, particles, velocity):
        rp, rg = np.random.uniform(size=2)

        velocity = velocity + self.a*(rp*(self.pbest-particles) + rg*(self.gbest-particles))
        particles = particles + velocity
        fitness = np.array(fit(particles))
        self.pbest[(fit(particles) >= fitness)] = particles[(fit(particles) >= fitness)]
        if min(fit(self.pbest)) <= fit(self.gbest):
            self.gbest = self.pbest[fit(particles).argmin()]

        return particles, velocity

if __name__ == '__main__':
    ga = GA()
    pso = PSO()
    
    for i in range(100):
        population = ga.create_population()
        particles, velocity = pso.generate()

        for i in range(100):
            population = ga.evolve(population)
            particles, velocity = pso.iterate(particles, velocity)
        
        data(ga.min_var, ga.min_fit, pso.gbest, fit(pso.gbest))
    plot()

