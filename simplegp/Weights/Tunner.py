from copy import deepcopy

from gaft import GAEngine
from gaft.analysis import FitnessStore
from gaft.components import IndividualBase, DecimalIndividual, Population
from gaft.operators import TournamentSelection, UniformCrossover, FlipBitMutation
from gaft.plugin_interfaces import OnTheFlyAnalysis

from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Nodes import BaseNode


class Tunner:

    def __init__(self, fitness: SymbolicRegressionFitness):
        self.individual = None
        self.fitness = fitness

    def set_individual(self, individual: BaseNode):
        self.individual = deepcopy(individual)

    def tuneWeights(self, range_scaling=(0.5, 1.5), range_translation=(-5, 5)):
        old_fitness = self.individual.fitness

        weights_scaling = self.individual.get_subtree_scaling()
        weights_translation = self.individual.get_subtree_translation()

        range = [range_scaling, ] * len(weights_scaling) + [range_translation, ] * len(weights_translation)
        indv_template = DecimalIndividual(ranges=range, eps=0.001)
        population = Population(indv_template=indv_template, size=26)
        population.init()

        engine = GAEngine(
            population=population,
            selection=TournamentSelection(),
            crossover=UniformCrossover(pc=1, pe=0.5),
            mutation=FlipBitMutation(pm=0.00000000001),
            fitness=self.fitnessFunction,
            analysis=[ConsoleOutputAnalysis]
        )
        engine.run(ng=20)

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
        self.fitness.evaluate(self.individual)

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
