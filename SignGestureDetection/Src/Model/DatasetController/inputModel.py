import os
import joblib
import numpy as np
from Constraints.path import PICKELS_PATH
from Model.modelEnum import Environment, Image
from Exception.modelException import EnvironmentException
from Exception.inputOutputException import PathDoesNotExistException


class InputModel:

    NUMBER_IMAGES_IN_REDUCED_PICKEL = 5

    def __init__(self):
        self.__train_data = None
        self.__test_data = None
        self.pickels_name = []
        self.base_pickle_src = f"{PICKELS_PATH}%s/%s_%s.pkl"

    def __read_data(self, environment):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        data = self.__get_init_data()

        for pickel_name in self.pickels_name:

            actual_pickel_data = self.__get_pickel_data(pickel_name, environment)

            data = self.__concat_pickels(data, actual_pickel_data)

        return data

    def combine_pickels_reducing_size(self, environment):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        data = self.__get_init_data()

        for pickel_name in self.pickels_name:
            actual_pickel_data = self.__get_pickel_data(pickel_name, environment)
            actual_pickel_data = self.__get_firsts_values_each_sign(actual_pickel_data)
            data = self.__concat_pickels(data, actual_pickel_data)

        if environment is Environment.TRAIN:
            self.__train_data = data
        else:
            self.__test_data = data

    def __get_firsts_values_each_sign(self, actual_pickel_data):
        data = self.__get_init_data()
        data[Image.DESCRIPTION.value] = actual_pickel_data[Image.DESCRIPTION.value]

        signs = np.unique(actual_pickel_data[Image.LABEL.value])

        for sign in signs:
            indexes = [index for index, value in enumerate(actual_pickel_data[Image.LABEL.value]) if value == sign][:self.NUMBER_IMAGES_IN_REDUCED_PICKEL]

            if len(data[Image.DATA.value]) == 0:
                data[Image.DATA.value] = actual_pickel_data[Image.DATA.value][indexes]
                data[Image.LABEL.value] = actual_pickel_data[Image.LABEL.value][indexes]
            else:
                data[Image.DATA.value] = np.concatenate((
                    data[Image.DATA.value],
                    actual_pickel_data[Image.DATA.value][indexes]
                ))
                data[Image.LABEL.value] = np.concatenate((
                    data[Image.LABEL.value],
                    actual_pickel_data[Image.LABEL.value][indexes]
                ))

        return data

    @staticmethod
    def __concat_pickels(data, actual_pickel_data):
        if len(data[Image.DATA.value]) == 0:
            data[Image.DATA.value] = actual_pickel_data[Image.DATA.value]
            data[Image.LABEL.value] = actual_pickel_data[Image.LABEL.value]
            data[Image.DESCRIPTION.value] = actual_pickel_data[Image.DESCRIPTION.value]

        else:
            data[Image.DATA.value] = np.concatenate((
                data[Image.DATA.value],
                actual_pickel_data[Image.DATA.value]
            ))
            data[Image.LABEL.value] = np.concatenate((
                data[Image.LABEL.value],
                actual_pickel_data[Image.LABEL.value]
            ))
            data[Image.DESCRIPTION.value] += "; " + actual_pickel_data[Image.DESCRIPTION.value]

        return data

    def __get_pickel_data(self, pickel_name, environment):
        pickle_src = self.base_pickle_src % (pickel_name, pickel_name, environment.value)

        if not os.path.exists(pickle_src):
            raise PathDoesNotExistException("The pickle needs to exists before using it")

        actual_pickel_data = joblib.load(pickle_src)
        actual_pickel_data[Image.DATA.value] = np.array(actual_pickel_data[Image.DATA.value])
        actual_pickel_data[Image.LABEL.value] = np.array(actual_pickel_data[Image.LABEL.value])

        return actual_pickel_data

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

    def set_pickels_name(self, pickels_name):
        self.pickels_name = pickels_name

    def get_pickels_name(self):
        return "-".join(self.pickels_name)
