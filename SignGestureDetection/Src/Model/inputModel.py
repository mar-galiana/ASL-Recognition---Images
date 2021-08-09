import os
import joblib
import numpy as np
from path import PICKELS_PATH
from Model.enumerations import Environment, Image
from Exception.modelException import EnvironmentException
from Exception.inputOutputException import PathDoesNotExistException


class InputModel:

    def __init__(self):
        self.__train_data = None
        self.__test_data = None
        self.pickels_name = []
        self.base_pickle_src = f"{PICKELS_PATH}%s/%s_%s.pkl"

    def __read_data(self, environment):
        data = {
            Image.DESCRIPTION.value: "",
            Image.DATA.value: list(),
            Image.LABEL.value: list()
        }

        for pickel_name in self.pickels_name:

            pickle_src = self.base_pickle_src % (pickel_name, pickel_name, environment.value)

            if not os.path.exists(pickle_src):
                raise PathDoesNotExistException("The pickle needs to exists before using it")

            actual_pickel_data = joblib.load(pickle_src)

            if len(data[Image.DATA.value]) == 0:
                data[Image.DATA.value] = actual_pickel_data[Image.DATA.value]
                data[Image.LABEL.value] = actual_pickel_data[Image.LABEL.value]
                data[Image.DESCRIPTION.value] = actual_pickel_data[Image.DESCRIPTION.value]

            else:
                data[Image.DATA.value] = np.concatenate((data[Image.DATA.value], actual_pickel_data[Image.DATA.value]))
                data[Image.LABEL.value] = np.concatenate((data[Image.LABEL.value], actual_pickel_data[Image.LABEL.value]))
                data[Image.DESCRIPTION.value] += "; " + actual_pickel_data[Image.DESCRIPTION.value]

        return data

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
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if environment == Environment.TRAIN:
            self.__train_data[Image.DATA.value] = data
        else:
            self.__test_data[Image.DATA.value] = data

    def set_y(self, environment, label):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if environment == Environment.TRAIN:
            self.__train_data[Image.LABEL.value] = label
        else:
            self.__test_data[Image.LABEL.value] = label

    def set_pickels_name(self, pickels_name):
        self.pickels_name = pickels_name

    def get_pickels_name(self):
        return "-".join(self.pickels_name)
