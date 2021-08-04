import os
import numpy as np
from Model.enumerations import Image
from Model.inputModel import InputModel
from Model.outputModel import OutputModel


class Model:

    def __init__(self, width=150, height=None):
        self.output_model = OutputModel(width, height)
        self.input_model = InputModel()

    def create_pickle(self, pickel_name, dataset, environments_separated, as_gray):
        self.output_model.create_pickle(pickel_name, dataset, environments_separated, as_gray)

    def set_pickel_name(self, name):
        self.input_model.set_pickel_name(name)

    def get_pickel_name(self):
        return self.input_model.get_pickel_name()

    def get_x(self, environment):
        data = self.input_model.get_data(environment)
        return np.array(data[Image.DATA.value])

    def get_y(self, environment):
        data = self.input_model.get_data(environment)
        return np.array(data[Image.LABEL.value])

    def set_x(self, environment, data):
        self.input_model.set_x(environment, data)

    def set_y(self, environment, label):
        self.input_model.set_y(environment, label)
