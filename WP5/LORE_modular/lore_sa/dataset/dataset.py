__all__ = ["Dataset"]

from abc import abstractmethod

from lore_sa.logger import logger


class Dataset():
    """
    Generic class to handle datasets
    """
    @abstractmethod
    def update_descriptor(self):
        """
        it creates the dataset descriptor dictionary
        """

    def set_target_label(self, descriptor):
        """
        Set the target column into the dataset descriptor

        :param descriptor:
        :return: a modified version of the input descriptor with a new key 'target'
        """
        if self.class_name is None:
            logger.warning("No target class is defined")
            return descriptor

        for type in descriptor:
            for k in descriptor[type]:
                if k == self.class_name:
                    descriptor['target'] = {k: descriptor[type][k]}
                    descriptor[type].pop(k)
                    return descriptor

        return descriptor


    def set_descriptor(self, descriptor):
        self.descriptor = descriptor
        self.descriptor = self.set_target_label(self.descriptor)

    def set_class_name(self,class_name: str):
        """
        Set the class name. Only the column name string
        :param [str] class_name:
        :return:
        """
        self.class_name = class_name
        self.descriptor = self.set_target_label(self.descriptor)

    def get_class_values(self):
        """
        return the list of values of the target column
        :return:
        """
        if self.class_name is None:
            raise Exception("ERR: class_name is None. Set class_name with set_class_name('<column name>')")
        print("test1", self.descriptor['target'])
        return self.descriptor['target'][self.class_name]['distinct_values']

    def get_numeric_columns(self):
        numeric_columns = list(self.descriptor['numeric'].keys())
        return numeric_columns

    def get_categorical_columns(self):
        categorical_columns = list(self.descriptor['categorical'].keys())
        return categorical_columns

    def get_feature_names(self):
        return self.get_numeric_columns() + self.get_categorical_columns()

    def get_number_of_features(self):
        return len(self.get_feature_names())

    def get_feature_name(self, index):
        pass

    def get_feature_name(self, index):
        """
        Get the feature name by index
        :param index:
        :return: the name of the corresponding feature
        """
        for category in self.descriptor.keys():
            for name in self.descriptor[category].keys():
                if self.descriptor[category][name]['index'] == index:
                    return name