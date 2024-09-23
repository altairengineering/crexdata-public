from lore_sa.bbox import AbstractBBox
from lore_sa.dataset import Dataset
from lore_sa.encoder_decoder import EncDec
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator

from lore_sa.dataset.tabular_dataset import TabularDataset
import pandas as pd
import numpy as np

__all__ = ["NeighborhoodGenerator", "RandomGenerator"]


class RandomGenerator(NeighborhoodGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from the instance to explain
    """

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1):
        """
        :param bbox: the Black Box model to explain
        :param dataset: the dataset with the descriptor of the original dataset
        :param encoder: an encoder to transfrom the data from/to the black box model
        :param ocr: acronym for One Class Ratio, it is the ratio of the number of instances of the minority class
        """
        super().__init__(bbox, dataset, encoder, ocr)
        self.generated_data = None


    def generate(self, x, num_instances, descriptor, encoder):
        """
        random generation of new instances. The starting instance x is only used to detect the value type of each feature, in order
        to generate new values only for numeric features.

        :param x[dict]: the starting instance from the real dataset
        :param num_instances[int]: the number of instances to generate
        :param descriptor[dict]: data descriptor as generated from a Dataset object
        :param onehotencoder[EncDec]: the onehotencoder eventually to encode the instance
        The list (or range) associated to each key is used to randomly choice an element within the list.

        :return [TabularDataset]: a tabular dataset instance with the new data generated
        """

        generated_list = np.array([])
        columns = np.empty(len(x), dtype=object)

        for n in range(num_instances):
            instance = np.empty(len(x), dtype=object)

            # TODO: needs refactoring to improve efficiency
            for name, feature in descriptor['categorical'].items():
                if encoder is not None:
                    # feature is encoded, so i need to random generate chunks of one-hot-encoded values

                    # finding the vector index of the feature
                    indices = [k for k, v in encoder.get_encoded_features().items() if v.split("=")[0] == name]
                    index_choice = np.random.choice(list(range(len(indices))))

                    for i, idx in enumerate(indices):
                        if i == index_choice:
                            instance[idx] = 1
                        else:
                            instance[idx] = 0
                        columns[idx] = encoder.get_encoded_features()[idx]


                else:
                    # feature is not encoded: random choice among the distinct values of the feature

                    instance[feature['index']] = np.random.choice(feature['distinct_values'])
                    columns[feature['index']] = name

            for name, feature in descriptor['numeric'].items():
                idx = None
                if encoder is not None:
                    idx = [k for k, v in encoder.get_encoded_features().items() if v == name][0]
                else:
                    idx = feature['index']
                columns[idx] = name

                instance[idx] = np.random.uniform(low=feature['min'], high=feature['max'])

            # append instance array to generated_list
            if generated_list.size == 0:
                generated_list = instance
            else:
                generated_list = np.vstack((generated_list, instance))

        balanced_list = super().balance_neigh(x, generated_list, num_instances)
        self.generated_data = balanced_list

        return balanced_list

