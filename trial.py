import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib
import sklearn
individual_char = ['rand_cell_type', 'rand_test', 'rand_material', 'rand_time', 'rand_conc', 'viability', 'rand_hd', 'rand_zeta']
population_size = 50    # number of individuals in generation
max_generations = 50    # maximal number of generations
elite_group = 5 # top 5 compounds which have highest fitness score after random feature (they are not exposed to mutation and crossover)
desired_fitness = 0.9

prob_crossover = 0.8    # probability of each gene exchange
prob_mutation = 0.1

db = pd.read_csv('Database/Cytotoxicity.csv')   # here the toxicity database is loaded
#trained_model = joblib.load('ML_models/Trained_model.joblib')

class Individual: # create a random individual with random characteristics that consist feature similar to original data
    def __init__(self):
        cell_type = db['Cell type']
        test = db['test']
        material = db['material']
        time = db['time (hr)']
        concentration = db['concentration (ug/ml)']
        viability = db['viability (%)']  # to be determined by the trained model
        hd = db['Hydrodynamic diameter (nm)']
        zeta = db['Zeta potential (mV)']
        rand_cell_type = random.choice(cell_type)
        rand_test = random.choice(test)
        rand_material = random.choice(material)
        rand_time = random.choice(time)
        rand_conc = random.choice(concentration)
        rand_hd = random.choice(hd)
        rand_zeta = random.choice(zeta)
        self.individual = [rand_cell_type, rand_test, rand_material, rand_time, rand_conc, viability, rand_hd, rand_zeta]
        self.fitness = 0

    def get_individual(self): # return the individual feature for further processing
        return self.individual

    def get_fitness(self):  # return the fitness for further processing
        self.fitness = 0
        self.fitness = Individual.get_individual()[6]
        # modify this script to get fitness values



        return self.fitness

    def __str__(self):  # to print the individual feature and fitness
        return self.individual.__str__()
        #return 'Individual:' + str(self.individual) + 'Fitness:' + str(self.fitness)


class Population:
    def __init__(self, size):
        self._populations = []
        i = 0
        while i < size:
            self._populations.append(Individual()) #it will join all generated nanomaterial into one single array with all its feature
        i += 1

    def get_individual(self):
        return self._populations

class GeneticAlgorithm:
    @staticmethod
    def evolve(feature):
        return GeneticAlgorithm.mutate_features(GeneticAlgorithm.crossover_features(feature))

    @staticmethod
    def crossover_features(feature):
        crossover_char = Population(0) #
        for i in range(elite_group):
            crossover_char.get_individual().append(feature.get_population()[i])
        i = elite_group
        while i < population_size:
            individual1 = GeneticAlgorithm.select_tournament_individuals(feature).get_population()[0]
            individual2 = GeneticAlgorithm.select_tournament_individuals(feature).get_population()[0]
            crossover_char.get_individual().append(GeneticAlgorithm.crossover_individuals(individual1, individual2))
            i += 1
        return crossover_char

    @staticmethod
    def mutate_features(feature):
        for i in range(elite_group, population_size):
            GeneticAlgorithm.mutate_individual(feature.get_population()[i])
            return feature

    @staticmethod
    def crossover_individuals(individual1, individual2):
        cross_individual = Individual()
        for i in range(individual_char.__len__()):
            if random.random() < prob_crossover:
                if random.random() >= 0.5:
                    cross_individual.get_individual()[i] = individual1.get_individual()[i]
                else:
                    cross_individual.get_individual()[i] = individual2.get_individual()[i]
            return cross_individual
    @staticmethod
    def mutate_individual(char):
        for i in range (individual_char.__len__()):
            if random.random() < prob_mutation:
                if random.random() < 0.5: #code might not work- here change the feature based on the list and position
                    char.get_individual()[i] = random.choice()[i]
                else:
                    char.get_individual()[i] = random.choice()[i]

def print_all_feature(feature, generation):
    print("\n ........")
    print("Generation Number: ", generation, "| Best fitness Features: ", feature.get_individual()[0].get_fitness())
    print(".........")
    i = 0
    for x in feature.get_individual():
        print("Individual#", i, " :", x, "| Fitness: ", x.get_fitness())
        i += 1

population = Population(population_size)
population.get_individual().sort(key=lambda x:x.get_fitness(), reverse=True)
print_all_feature(population, 0)
generation_number = 1
while population.get_individual()[0].get_fitness() < individual_char.__len__():
    Population = GeneticAlgorithm.evolve(Population)
    Population.get_individual().sort(key=lambda x: x.get_fitness(), reverse=True)
    print_all_feature(Population, generation_number)
    generation_number += 1


