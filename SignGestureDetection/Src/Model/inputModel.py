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
        self.pickel_name = "sign_gesture"
        self.base_pickle_src = f"{PICKELS_PATH}{self.pickel_name}/{self.pickel_name}_%s.pkl"

    def __read_data(self, environment):

        pickle_src = self.base_pickle_src % environment.value
        print(pickle_src)

        if os.path.exists(pickle_src):
            data = joblib.load(pickle_src)
        else:
            raise PathDoesNotExistException("The pickle needs to exists before using it hh")

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

    def set_pickel_name(self, pickel_name):
        self.pickel_name = pickel_name
        self.base_pickle_src = f"{PICKELS_PATH}{pickel_name}/{pickel_name}_%s.pkl"

    def get_pickel_name(self):
        return self.pickel_name
