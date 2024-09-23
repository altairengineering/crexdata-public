from abc import ABC, abstractmethod

__all__ = ["Surrogate"]

import numpy as np

from lore_sa.dataset import Dataset
from lore_sa.encoder_decoder import EncDec


class Surrogate(ABC):
    """
    Generic surrogate class
    """
    def __init__(self, kind = None, preprocessing =None):
        #decision tree, supertree
        self.kind = kind
        #kind of preprocessing to apply
        self.preprocessing = preprocessing

    @abstractmethod
    def train(self, Z, Yb, weights):
        pass

    @abstractmethod
    def get_rule(self, x: np.array, encdec: EncDec = None):
        pass

    @abstractmethod
    def get_counterfactual_rules(self, x: np.array, neighborhood_train_X: np.array, neighborhood_train_Y: np.array,
                                 encoder: EncDec = None,
                                 filter_crules=None, constraints: dict = None, unadmittible_features: list = None):
        pass