from copy import deepcopy

from gaft import GAEngine
from gaft.analysis import FitnessStore
from gaft.components import IndividualBase, DecimalIndividual, Population
from gaft.operators import TournamentSelection, UniformCrossover, FlipBitMutation
from gaft.plugin_interfaces import OnTheFlyAnalysis

from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Nodes import BaseNode


class Tuner:

    def __init__(self, fitness_function=None,
                 scale_range=(-5, 5), translation_range=(-5, 5),
                 run_generations=(),
                 population_fraction=1,
                 max_iterations=20,
                 pop_size=100):

        """
        Weight tuner for the variables of the real valued GA
        :param fitness_function: fitness function to apply
        :param scale_range: tuple that indicates the range that the scaling variables can take on. Default: (-5, 5)
        :param translation_range: tuple that indicates the range that the translation variables can take on. Default: (-5, 5)
        :param run_generations: tuple containing the generation numbers to apply weight tuning e.g (99, 100).
        Defaults to not running in any generation i.e an empty tuple.
        :param population_fraction: probability that tuning is applied to an individual in any given generation
        e.g 0.5 for half of the population. Defaults to all individuals i.e. 1.
        :param max_iterations: Number of iterations to run the real valued GA for. Default 20.
        :param pop_size: Population size to be used by the real valued GA. Default 100
        """
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.population_fraction = population_fraction
        self.run_generations = run_generations
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.individual = None
        self.fitness_function = fitness_function

    def set_individual(self, individual: BaseNode):
        self.individual = deepcopy(individual)

    def tuneWeights(self):
        old_fitness = self.individual.fitness

        weights_scaling = self.individual.get_subtree_scaling()
        weights_translation = self.individual.get_subtree_translation()
        # Create array with range for each scaling and translation parameter
        range = [self.scale_range, ] * len(weights_scaling) + [self.translation_range, ] * len(weights_translation)
        indv_template = DecimalIndividual(ranges=range, eps=0.001)
        population = Population(indv_template=indv_template, size=self.pop_size)
        population.init()

        engine = GAEngine(
            population=population,
            selection=TournamentSelection(),
            crossover=UniformCrossover(pc=1, pe=0.5),
            mutation=FlipBitMutation(pm=0.00000000001),
            fitness=self.fitnessFunction,
            analysis=[ConsoleOutputAnalysis]
        )

        # Run the GA with the specified number of iterations
        engine.run(ng=self.max_iterations)

        # Get the best individual.
        best_indv = engine.population.best_indv(engine.fitness)
        if old_fitness > -engine.ori_fmax:
            weights_scaling, weights_translation = self.split_list(best_indv.solution)

            self.individual.set_subtree_scaling(weights_scaling)
            self.individual.set_subtree_translation(weights_translation)

        return deepcopy(self.individual)

    def fitnessFunction(self, base: IndividualBase):
        weights_scaling, weights_translation = self.split_list(base.solution)

        self.individual.set_subtree_scaling(weights_scaling)
        self.individual.set_subtree_translation(weights_translation)
        self.fitness_function.evaluate(self.individual)

        return -self.individual.fitness

    def split_list(self, a_list):
        half = len(a_list) // 2

        return a_list[:half], a_list[half:]


class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 10
    master_only = True

    def finalize(self, population, engine):
        y = engine.ori_fmax
        msg = 'Optimal solution: {}'.format(-y)
        self.logger.info(msg)
