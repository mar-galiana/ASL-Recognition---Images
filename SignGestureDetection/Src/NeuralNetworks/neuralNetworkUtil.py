import os
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from Model.enumerations import Environment
from NeuralNetworks.enumerations import NeuralNetworkEnum
from Exception.modelException import EnvironmentException


class NeuralNetworkUtil:

    def __init__(self, model):
        self.KERAS_MODEL_BASE_PATH = f"{os.getcwd()}/../Assets/NeuralNetworkModel/"
        self.model = model

    def train_model(self, sequential_model):
        # looking at the model summary
        sequential_model.summary()

        # compiling the sequential model
        sequential_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # training the model for 10 epochs
        sequential_model.fit(self.model.get_x(Environment.TRAIN),
                             self.model.get_y(Environment.TRAIN),
                             batch_size=128,
                             epochs=10,
                             validation_data=(self.model.get_x(Environment.TEST), self.model.get_y(Environment.TEST)))

        return sequential_model

    def save_keras_model(self, sequential_model, neural_network_type):
        sequential_model.save(self.__get_keras_model_path(neural_network_type))

    def load_keras_model(self, neural_network_type):
        return keras.models.load_model(self.__get_keras_model_path(neural_network_type))

    def __get_keras_model_path(self, neural_network_type):
        if not isinstance(neural_network_type, NeuralNetworkEnum):
            raise EnvironmentException("Environment used is not a valid one")

        if NeuralNetworkEnum.CNN == neural_network_type:
            file_name = f"cnn_{self.model.get_pickel_name()}_model"
        else:
            file_name = f"nn_{self.model.get_pickel_name()}_model"

        return self.KERAS_MODEL_BASE_PATH + file_name + ".h5"
