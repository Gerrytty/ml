import random
import numpy as np

N = 10
MAX = 20

def generate_problem():
    coefs = np.random.randint(MAX, size=N)
    ans = random.randint(0, MAX * (MAX // 2))
    return coefs, ans


class Individual:
    def __init__(self, gen=None, fitness=0):
        self.fitness = fitness
        if gen is None:
            self.gen = []
            for i in range(N):
                self.gen.append(random.randint(0, 20))
        else:
            self.gen = gen

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness


class Population:

    def __init__(self, individuals=None):
        if individuals is None:
            self.individuals = []
        else:
            self.individuals = individuals

    def init_population(self, population_size=20):
        self.individuals = [Individual() for _ in range(population_size)]

    def get_bests(self):
        return sorted(self.individuals)[:5]


class Mix:
    @staticmethod
    def simple_crossover(individual1: Individual, individual2: Individual):
        crossover_point = random.randint(0, N)
        return Individual(gen=(individual1.gen[:crossover_point] + individual2.gen[crossover_point:]))

    @staticmethod
    def simple_mutation(individual: Individual, mutation_count=2):
        new_gen = []

        for i in range(N):
            new_gen.append(individual.gen[i])

        for i in range(mutation_count):
            mutation_index = random.randint(0, N - 1)
            new_gen[mutation_index] = random.randint(0, MAX)

        return Individual(gen=new_gen)


class Evolution:

    @staticmethod
    def get_fitness(individual: Individual):

        individual_ans = 0

        for i in range(N):
            individual_ans += individual.gen[i] * coefs[i]

        individual.fitness = (individual_ans - ans) ** 2

        if individual.fitness == 0:
            print(f"eqution was {coefs}; ans {ans}; solution {individual.gen}")
            return True

        else:
            return False

    @staticmethod
    def calc(pop):
        for individual in pop.individuals:
            if Evolution.get_fitness(individual):
                return True, None

        bests = pop.get_bests()

        news = []
        for best in bests:
            news.append(best)
        for i in range(3):
            i1_index = random.randint(0, len(pop.individuals) - 1)
            i2_index = random.randint(0, len(pop.individuals) - 1)
            i3_index = random.randint(0, len(pop.individuals) - 1)
            news.append(Mix.simple_crossover(pop.individuals[i1_index], pop.individuals[i2_index]))
            news.append(Mix.simple_mutation(pop.individuals[i3_index]))

        return False, Population(individuals=news)


if __name__ == "__main__":
    coefs, ans = generate_problem()

    first_population = Population()
    first_population.init_population()

    i1 = first_population.individuals[0]
    i2 = first_population.individuals[1]
    i = 0
    while True:
        solution_is_find, first_population = Evolution.calc(first_population)
        if (solution_is_find):
            break

        print(f"Iter {i}")
        i += 1
        print("---------------------------")