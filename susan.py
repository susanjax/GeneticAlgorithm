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
        self.fitness = Individual.get_individual()
        return self.fitness

def print_fitness (fit):
    print('Fitness:', fit.Individual.get_individual()[6].get_fitness())

print_fitness(Individual.get_individual())