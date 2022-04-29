import random
INDIVIDUAL_NUMBER = 15
ELITE_POPULATION = 2
Tournament_selection = 4
Mutation_rate = 0.1
Desire_character = [ 1, 0, 1, 1, 0, 1, 0, 1, 0, 1,]

class Individual:
    def __init__(self):
        self.character = []
        self.fitness = 0
        i = 0
        while i < Desire_character.__len__():
            if random.random() >= 0.5:
                self.character.append(1)
            else:
                self.character.append(0)
            i += 1

    def get_character(self):
        return self.character

    def get_fitness(self):
        self.fitness = 0
        for i in range(self.character.__len__()):
            if self.character[i] == Desire_character[i]:
                self.fitness += 1
        return self.fitness

    def __str__(self):
        return self.character.__str__()

class people:
    def __init__(self, size):
        self._individuals = []
        i = 0
        while i < size:
            self._individuals.append(Individual())
            i += 1

    def get_individuals(self): #for chromosome
        return self._individuals


class GeneticAlgorithm:
    @staticmethod
    def evolve(feature):
        return GeneticAlgorithm.mutate_features(GeneticAlgorithm.crossover_features(feature))

    @staticmethod
    def crossover_features(feature):
        crossover_char = people(0)
        for i in range (ELITE_POPULATION):
            crossover_char.get_individuals().append(feature.get_individuals()[i])
        i = ELITE_POPULATION
        while i < INDIVIDUAL_NUMBER:
            individual1 = GeneticAlgorithm.select_tournament_individuals(feature).get_individuals()[0]
            individual2 = GeneticAlgorithm.select_tournament_individuals(feature).get_individuals()[0]
            crossover_char.get_individuals().append(GeneticAlgorithm.crossover_individuals(individual1, individual2))
            i += 1
        return crossover_char

    @staticmethod
    def mutate_features(feature):
        for i in range(ELITE_POPULATION, INDIVIDUAL_NUMBER):
            GeneticAlgorithm.mutate_individual(feature.get_individuals()[i])
            return feature

    @staticmethod
    def crossover_individuals(individual1, individual2):
        cross_individual = Individual()
        for i in range (Desire_character.__len__()):
            if random.random() >= 0.5:
                cross_individual.get_character()[i] = individual1.get_character()[i]
            else:
                cross_individual.get_character()[i] = individual2.get_character()[i]
            return cross_individual

    @staticmethod
    def mutate_individual(char):
        for i in range (Desire_character.__len__()):
            if random.random() < Mutation_rate:
                if random.random() < 0.5:
                    char.get_character()[i] = 1
                else:
                    char.get_character()[i] = 0

    @staticmethod
    def select_tournament_individuals(feature):
        tournament_individual = people(0)
        i = 0
        while i < Tournament_selection:
            tournament_individual.get_individuals().append(feature.get_individuals()[random.randrange(0, INDIVIDUAL_NUMBER)])
            i += 1
        tournament_individual.get_individuals().sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_individual

def print_all_feature(feature, generation):
    print("\n ........")
    print("Generation Number: ", generation, "| Best fitness Features: ", feature.get_individuals()[0].get_fitness())
    print("Desire character: ", Desire_character )
    print(".........")
    i = 0
    for x in feature.get_individuals():
        print("Invididual#", i, " :", x, "| Fitness: ", x.get_fitness())
        i += 1

Population = people(INDIVIDUAL_NUMBER)
Population.get_individuals().sort(key=lambda x:x.get_fitness(), reverse=True)
print_all_feature(Population, 0)
generation_number = 1
while Population.get_individuals()[0].get_fitness() < Desire_character.__len__():
    Population = GeneticAlgorithm.evolve(Population)
    Population.get_individuals().sort(key=lambda x: x.get_fitness(), reverse=True)
    print_all_feature(Population, generation_number)
    generation_number += 1


