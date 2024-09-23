import numbers
import pickle
import random

from scipy.spatial.distance import cdist, hamming
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from deap import base, creator, tools, algorithms

import numpy as np

__all__ = ["NeighborhoodGenerator", "GeneticGenerator"]

from lore_sa.util import neuclidean, sigmoid


class GeneticGenerator(NeighborhoodGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from an input instance and
    pruning the generation around a fitness function based on proximity to the instance to explain
    """
    def __init__(self, bbox=None, dataset=None, encoder=None, ocr=0.1,
                 alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None):
        """

        :param bbox: the Black Box model to explain
        :param dataset: the dataset with the descriptor of the original dataset
        :param encoder: an encoder to transfrom the data from/to the black box model
        :param ocr: acronym for One Class Ratio, it is the ratio of the number of instances of the minority class
        :param alpha1: the weight of the similarity of the features from the given instance. The sum of alpha1 and alpha2 must be 1
        :param alpha2: the weight of the similiarity of the target class from the given instance. The sum of alpha1 and alpha2 must be 1
        :param metric: the distance metric to use to compute the distance between instances
        :param ngen: the number of generations to run
        :param mutpb: probability of mutation of a specific feature
        :param cxpb:
        :param tournsize:
        :param halloffame_ratio:
        :param random_seed: initial seed for the random number generator
        """
        self.bbox = bbox
        self.dataset = dataset
        self.encoder = encoder
        self.ocr = ocr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.tournsize = tournsize
        self.halloffame_ratio = halloffame_ratio
        self.random_seed = random_seed
        random.seed(random_seed)

    def generate(self, z, num_instances, descriptor, encoder):
        """
        The generation is based on the strategy of generating a number of instances for the same class as the input
        instance and a number of instances for a different class.
        The generation of the instances for each subgroup is done through a genetic algorithm based on two fitness
        fuctions: one for the same class and one for the different class.
        :param z: the input instance
        :param num_instances: how many elements to generate
        :param descriptor: the descriptor of the dataset
        :return:
        """
        new_x = z.copy()

        # determine the number of instances to generate for the same class and for a different class
        num_samples_eq = int(np.round(num_instances * 0.5))
        num_samples_neq = num_instances - num_samples_eq

        # generate the instances for the same class
        toolbox_eq = self.setup_toolbox(z, self.fitness_equal, num_samples_eq)
        population_eq, halloffame_eq, logbook_eq = self.fit(toolbox_eq, num_samples_eq)
        Z_eq = self.add_halloffame(population_eq, halloffame_eq)
        # print(logbook_eq)

        # generate the instances for a different class
        toolbox_noteq = self.setup_toolbox(z, self.fitness_notequal, num_samples_neq)
        population_noteq, halloffame_noteq, logbook_noteq = self.fit(toolbox_noteq, num_samples_neq)
        Z_noteq = self.add_halloffame(population_noteq, halloffame_noteq)
        # print(logbook_noteq)

        # concatenate the two sets of instances
        Z = np.concatenate((Z_eq, Z_noteq), axis=0)

        # balance the instances according to the minority class
        Z = super(GeneticGenerator, self).balance_neigh(z, Z, num_instances)
        # the first element is the input instance

        Z[0] = new_x
        return Z

    def add_halloffame(self, population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i + 1] - fitness_values[i] for i in range(0, len(fitness_values) - 1)]

        sorted_array = np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist()
        if len(sorted_array) == 0:
            fitness_value_thr = -np.inf
        else:
            index = np.max(sorted_array)
            fitness_value_thr = fitness_values[index]

        Z = list()
        for p in population:
            # if p.fitness.wvalues[0] > fitness_value_thr:
            Z.append(p)

        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                Z.append(h)

        return np.array(Z)

    def setup_toolbox(self, x, evaluate, population_size):

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        return toolbox

    def setup_toolbox_noteq(self, x, x1, evaluate, population_size):

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x1)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        return toolbox

    def fit(self, toolbox, population_size):

        halloffame_size = int(np.round(population_size * self.halloffame_ratio))

        population = toolbox.population(n=population_size)
        halloffame = tools.HallOfFame(halloffame_size, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                                  ngen=self.ngen, stats=stats, halloffame=halloffame,
                                                  verbose=True)

        return population, halloffame, logbook

    def record_init(self, x):
        return x

    def random_init(self):
        z = self.generate_synthetic_instance()
        x = self.encoder.decode(z.reshape(1, -1))
        if None in x :
            print('None in generated z')
            print('z', z)
            print('x', x)

        return z

    def clone(self, x):
        return pickle.loads(pickle.dumps(x))

    def mutate(self, toolbox, x):
        z = toolbox.clone(x)
        # for i in range(self.nbr_features):
        #         #     if np.random.random() <= self.mutpb:
        #         #         z[i] = np.random.choice(self.feature_values[i], size=1, replace=True)
        z = self.generate_synthetic_instance(from_z=z, mutpb=self.mutpb)
        x = self.encoder.decode(z.reshape(1, -1))
        if None in x :
            print('None in mutated z')
            print('z', z)
            print('x', x)
        return z,

    def fitness_equal(self, z, z1):
        if isinstance(self.metric, numbers.Number):
            self.metric = neuclidean
        feature_similarity_score = 1.0 - cdist(z.reshape(1, -1), z1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0
        # feature_similarity = sigmoid(feature_similarity_score)

        # y = self.bb_predict(x.reshape(1, -1))[0]
        # y1 = self.bb_predict(x1.reshape(1, -1))[0]
        x = self.encoder.decode(z.reshape(1, -1))
        x1 = self.encoder.decode(z1.reshape(1, -1))
        if None in x or None in x1:
            return 0.0, # TODO: check if this is the correct way to return a tuple

        y = self.bbox.predict(x)
        y1 = self.bbox.predict(x1)

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score >= self.eta2 else 0.0
        target_similarity = sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal(self, z, z1):
        feature_similarity_score = 1.0 - cdist(z.reshape(1, -1), z1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score)

        # y = self.bb_predict(x.reshape(1, -1))[0]
        # y1 = self.bb_predict(x1.reshape(1, -1))[0]
        x = self.encoder.decode(z.reshape(1, -1))
        x1 = self.encoder.decode(z1.reshape(1, -1))
        if None in x or None in x1:
            return 0.0, #TODO: check why we get here in the code
        y = self.bbox.predict(x)
        y1 = self.bbox.predict(x1)

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score < self.eta2 else 0.0
        target_similarity = 1.0 - sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,