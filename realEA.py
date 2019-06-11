import random

import numpy

from deap import base
from deap import creator
from deap import tools


class RealEA:
    def _init_(self):

        # Single objective Fitness class, minimizing fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Individual class with base type array (vector) with a fitness attribute set to jus tthe created fitness
        creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMin)

        # Equal to arity of the functions
        NUM_WEIGHTS = 2

        POP_SIZE = 10

        # Attribute generator
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_WEIGHTS)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

        # Operator registering
        toolbox.register("evaluate", evaluate)
        toolbox.register("crossover", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # TODO: Chose appropriate mutation type
        toolbox.register("select", tools.selTournament, tournsize=2)

        self.toolbox = toolbox

    def main(self):
        # TODO: initialize with all ones + some random

        pop = self.toolbox.population
        CROSS_PROB, M_PROB, MAX_GEN= 0.5,0.2, 20

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(MAX_GEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = map(self.toolbox.clone, offspring)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSS_PROB:
                    self.toolbox.crossover(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < M_PROB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring


        return pop


    # Evaluation function
    def evaluate(self, individual):
        return


def setWeights(gpIndividual):
    # arity of function nodes is 2, each argument has 2 weights, one constant and one scalar

    return None

# TODO: Allow the EA to run on the input of on individual
# TODO: Set max generations
# TODO: Somehow evaluate individuals (weights) using the fitness function of the GP individual
# TODO: Return GP individual with optimized weights
# TODO: Num weights 2* arity? Single scalar and single constant weight per function (see assignment) Or single scalar weight per argument (resulting in num_weights = arity.

# TODO: Main currently returns a population, how do we select a single winner?
# TODO: Use HallOfFame?
