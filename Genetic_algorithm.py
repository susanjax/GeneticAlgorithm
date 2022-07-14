import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib

cells = ['HepG2, Caco-2, PC12']    # these cells are chosen just as example

population_size = 50    # number of individuals in generation
max_generations = 50    # maximal number of generations
desired_fitness = 0.9

prob_crossover = 0.8    # probability of each gene exchange
prob_mutation = 0.1


db = pd.read_csv('Database/Cytotoxicity.csv')   # here the toxicity database is loaded
trained_model = joblib.load('ML_models/Trained_model.joblib')
# Defining all the classes

class FitnessMax:
  def __init__(self):
    self.values = [0]   # here the fitness value storage is defined

class Individual(list):
  def __init__(self, *args):
    super().__init__(*args)
    self.fitness = FitnessMax()    # here the fitness value of individual is stored

class population:
    def __init__(self, size):
    self.individuals =[]
    i = 0;
    while i < population_size
        self.individuals.append(Individual_creator())
        i += 1
# Defining all the functions

def one_max_fitness(nanomaterial, model=trained_model, cells_list=cells):

    """
    Inputs information about two cell types and
    one nanomaterial, as well as pre-trained model.

    cells - list with two elements, where the first
    is cell type to be killed, and the second is to
    survive.

    """

    nanomaterial_cell_1 = cells_list[0] + nanomaterial
    nanomaterial_cell_2 = cells_list[1] + nanomaterial

    viability_cell_1 = model.predict(nanomaterial_cell_1)
    viability_cell_2 = model.predict(nanomaterial_cell_2)

    final_fitness = viability_cell_2 / (viability_cell_1 + viability_cell_2)

    return final_fitness,


def individual_creator():
  
  """
  Creates a random nanoparticle by randomizing each 
  parameter based on the distribution in the database.
  
  If the size varies from 10 to 1000 nm, then random
  number from this range is chosen.
  
  If the parameter is not a number - then it is chosen
  randomly from the set presented in the database.
  
  Task:
  Your task here is to write a function, which will
  output the list similar in sequence to this in the
  database, but without cell characteristics and viability.
  """
  
  one_individual = Individual()
  
  return one_individual


def population_creator(n=0):
  return list([individual_creator() for _ in range(n)])


def clone(value):
  ind = Individual(value[:])
  ind.fitness.values[0] = value.fitness.values[0]
  return ind


def selection_tournament(whole_population, p_len):
    offspring_list = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

            offspring_list.append(max([whole_population[i1], whole_population[i2], whole_population[i3]],
                                 key=lambda ind: ind.fitness.values[0]))

    return offspring_list


def make_crossover(child_1, child_2):

  """
  All the categorical values should be interchanged with the
  defined probability.

  Numerical values are recalculated as mean value or weighted sum
  based on the fitness value.

  Task:
  This function takes two lists containing strings and numbers.
  It doesn't output anything but changes the inputted lists.
  """

  for gene_1, gene_2 in child_1, child_2:   # iterate over all list elements
    if isinstance(gene_1, str):    # check if string or not
        pass
    else:
        pass

# Genetic algorithm core

population = population_creator(n=population_size)
generation_count = 0

fitness_values = list(map(one_max_fitness, population))
for individual, fitness_value in zip(population, fitness_values):
  individual.fitness.values = fitness_value

max_fitness_values = []
mean_fitness_values = []

fitness_values = [individual.fitness.values[0] for individual in population]

total_individuals = population_size


while max(fitness_values) < desired_fitness and generation_count < max_generations:
  generation_count += 1
  offspring = selection_tournament(population, len(population))
  total_individuals += len(offspring)
  offspring = list(map(clone, offspring))

  for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < prob_crossover:
      make_crossover(child1, child2)

  freshFitnessValues = list(map(one_max_fitness, offspring))
  for individual, fitness_value in zip(offspring, freshFitnessValues):
    individual.fitness.values = fitness_value

  population[:] = offspring

  fitness_values = [ind.fitness.values[0] for ind in population]

  max_fitness = max(fitness_values)
  mean_fitness = sum(fitness_values) / len(population)
  max_fitness_values.append(max_fitness)
  mean_fitness_values.append(mean_fitness)
  print(f"Generation {generation_count}: Мax fitness = {max_fitness}, Mean fitness = {mean_fitness}")

  best_index = fitness_values.index(max(fitness_values))
  print("Best individual = ", *population[best_index], "\n")


plt.plot(max_fitness_values, color="red")
plt.plot(mean_fitness_values, color="green")
plt.xlabel("Generation")
plt.ylabel("Max/Mean fitness")
plt.xlim(0, 50)
plt.ylim(50, 101)
plt.title("Fitness dependence on the generation")
plt.show()