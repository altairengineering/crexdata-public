from lore_sa.bbox import AbstractBBox
import pandas as pd
import numpy as np

from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.encoder_decoder import ColumnTransformerEnc, EncDec
from lore_sa.neighgen import GeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.neighgen.random import RandomGenerator
from lore_sa.surrogate import DecisionTreeSurrogate, Surrogate


class Lore(object):

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec,
                 generator: NeighborhoodGenerator, surrogate: Surrogate):
        """
        Creates a new instance of the LORE method.


        :param bbox: The black box model to be explained wrapped in a ``AbstractBBox`` object.
        :param dataset:
        :param encoder:
        :param generator:
        :param surrogate:
        """

        super().__init__()
        self.bbox = bbox
        self.descriptor = dataset.descriptor
        self.encoder = encoder
        self.generator = generator
        self.surrogate = surrogate
        self.class_name = dataset.class_name


    def explain(self, x: np.array):
        """
        Explains a single instance of the dataset.
        :param x: an array with the values of the instance to explain (the target class is not included)
        :return:
        """
        # map the single record in input to the encoded space
        [z] = self.encoder.encode([x])
        # generate a neighborhood of instances around the projected instance `z`
        neighbour = self.generator.generate(z, 1000, self.descriptor, self.encoder)
        dec_neighbor = self.encoder.decode(neighbour)
        # split neighbor in features and class using train_test_split
        neighb_train_X = dec_neighbor[:, :]
        neighb_train_y = self.bbox.predict(neighb_train_X)
        neighb_train_yb = self.encoder.encode_target_class(neighb_train_y.reshape(-1, 1)).squeeze()

        # train the surrogate model on the neighborhood
        self.surrogate.train(neighbour, neighb_train_yb)

        # get the rule for the instance `z`, decode using the encoder class
        rule = self.surrogate.get_rule(z, self.encoder)
        # print('rule', rule)

        self.crules, self.deltas = self.surrogate.get_counterfactual_rules(z, neighbour, neighb_train_yb, self.encoder)

        return {'x': x, 'rule': rule, 'counterfactuals': self.crules, 'deltas': self.deltas}



class TabularRandomGeneratorLore(Lore):

        def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
            """
            Creates a new instance of the LORE method.


            :param bbox: The black box model to be explained wrapped in a ``AbstractBBox`` object.
            :param dataset:
            :param encoder:
            :param generator:
            :param surrogate:
            """
            encoder = ColumnTransformerEnc(dataset.descriptor)
            generator = RandomGenerator(bbox, dataset, encoder, 0.1)
            surrogate = DecisionTreeSurrogate()

            super().__init__(bbox, dataset, encoder, generator, surrogate)

        def explain_instance(self, x: np.array):
            return self.explain(x.values)

class TabularGeneticGeneratorLore(Lore):

        def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
            """
            Creates a new instance of the LORE method.


            :param bbox: The black box model to be explained wrapped in a ``AbstractBBox`` object.
            :param dataset:
            :param encoder:
            :param generator:
            :param surrogate:
            """
            encoder = ColumnTransformerEnc(dataset.descriptor)
            generator = GeneticGenerator(bbox, dataset, encoder, 0.1)
            surrogate = DecisionTreeSurrogate()

            super().__init__(bbox, dataset, encoder, generator, surrogate)

        def explain_instance(self, x: np.array):
            return self.explain(x.values)