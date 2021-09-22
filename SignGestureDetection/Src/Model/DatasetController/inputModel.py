import os
import joblib
import _pickle as cPickle
import gzip
import numpy as np
from Constraints.path import PICKLES_PATH
from Model.modelEnum import Environment, Image
from Exception.modelException import EnvironmentException
from Exception.inputOutputException import PathDoesNotExistException


class InputModel:

    NUMBER_IMAGES_IN_REDUCED_PICKLE = 5

    def __init__(self):
        self.__train_data = None
        self.__test_data = None
        self.pickles_name = []
        self.base_pickle_src = f"{PICKLES_PATH}%s/%s_%s.pkl"

    def __read_data(self, environment):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        data = self.__get_init_data()

        for pickle_name in self.pickles_name:

            actual_pickle_data = self.__get_pickle_data(pickle_name, environment)

            data = self.__concat_pickles(data, actual_pickle_data)

        return data

    def combine_pickles_reducing_size(self, environment):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        data = self.__get_init_data()

        for pickle_name in self.pickles_name:
            actual_pickle_data = self.__get_pickle_data(pickle_name, environment)
            actual_pickle_data = self.__get_firsts_values_each_sign(actual_pickle_data)
            data = self.__concat_pickles(data, actual_pickle_data)

        if environment is Environment.TRAIN:
            self.__train_data = data
        else:
            self.__test_data = data

    def __get_firsts_values_each_sign(self, actual_pickle_data):
        data = self.__get_init_data()
        data[Image.DESCRIPTION.value] = actual_pickle_data[Image.DESCRIPTION.value]

        signs = np.unique(actual_pickle_data[Image.LABEL.value])

        for sign in signs:
            indexes = [index for index, value in enumerate(actual_pickle_data[Image.LABEL.value]) if value == sign][:self.NUMBER_IMAGES_IN_REDUCED_PICKLE]

            if len(data[Image.DATA.value]) == 0:
                data[Image.DATA.value] = actual_pickle_data[Image.DATA.value][indexes]
                data[Image.LABEL.value] = actual_pickle_data[Image.LABEL.value][indexes]
            else:
                data[Image.DATA.value] = np.concatenate((
                    data[Image.DATA.value],
                    actual_pickle_data[Image.DATA.value][indexes]
                ))
                data[Image.LABEL.value] = np.concatenate((
                    data[Image.LABEL.value],
                    actual_pickle_data[Image.LABEL.value][indexes]
                ))

        return data

    @staticmethod
    def __concat_pickles(data, actual_pickle_data):
        if len(data[Image.DATA.value]) == 0:
            data[Image.DATA.value] = actual_pickle_data[Image.DATA.value]
            data[Image.LABEL.value] = actual_pickle_data[Image.LABEL.value]
            data[Image.DESCRIPTION.value] = actual_pickle_data[Image.DESCRIPTION.value]

        else:
            data[Image.DATA.value] = np.concatenate((
                data[Image.DATA.value],
                actual_pickle_data[Image.DATA.value]
            ))
            data[Image.LABEL.value] = np.concatenate((
                data[Image.LABEL.value],
                actual_pickle_data[Image.LABEL.value]
            ))
            data[Image.DESCRIPTION.value] += "; " + actual_pickle_data[Image.DESCRIPTION.value]

        return data

    def __get_pickle_data(self, pickle_name, environment):
        pickle_src = self.base_pickle_src % (pickle_name, pickle_name, environment.value)

        if not os.path.exists(pickle_src):
            raise PathDoesNotExistException("The pickle needs to exists before using it")

        with gzip.open(pickle_src, 'rb') as f:
            actual_pickle_data = cPickle.load(f)
        
        actual_pickle_data[Image.DATA.value] = np.array(actual_pickle_data[Image.DATA.value])
        actual_pickle_data[Image.LABEL.value] = np.array(actual_pickle_data[Image.LABEL.value])

        return actual_pickle_data

    @staticmethod
    def __get_init_data():
        return {
            Image.DESCRIPTION.value: "",
            Image.DATA.value: np.array([]),
            Image.LABEL.value: np.array([])
        }

    def get_data(self, environment):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if environment == Environment.TRAIN:
            if self.__train_data is None:
                self.__train_data = self.__read_data(environment)
            data = self.__train_data

        else:
            if self.__test_data is None:
                self.__test_data = self.__read_data(environment)
            data = self.__test_data

        return data

    def set_x(self, environment, data):
        self.__set_value(environment, Image.DATA, data)

    def set_y(self, environment, labels):
        self.__set_value(environment, Image.LABEL, labels)

    def __set_value(self, environment, data_type, values):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if self.__train_data is None:
            self.__train_data = {}

        if environment == Environment.TRAIN:
            self.__train_data[data_type.value] = values
        else:
            self.__test_data[data_type.value] = values

    def set_pickles_name(self, pickles_name):
        self.pickles_name = pickles_name

    def get_pickles_name(self):
        return "-".join(self.pickles_name)
