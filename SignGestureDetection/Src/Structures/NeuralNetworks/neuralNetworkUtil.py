import os
import numpy as np
from tensorflow import keras
from Model.enumerations import Environment
from Constraints.path import NEURAL_NETWORK_MODEL_PATH
from tensorflow.python.keras.utils import np_utils
from Exception.modelException import EnvironmentException
from Structures.iUtilStructure import IUtilStructure, Structure
from Structures.NeuralNetworks.enumerations import NeuralNetworkEnum
from Exception.inputOutputException import PathDoesNotExistException
from Src.Constraints.path import BINARY_CNN_MODEL_PATH, TMP_BINARY_CNN_MODEL_PATH


class NeuralNetworkUtil(IUtilStructure):

    def __init__(self, logger, model):
        self.logger = logger
        self.model = model

    def train_model(self, sequential_model, batch_size=128, epochs=10, is_categorical=False):
        loss = ('binary_crossentropy', 'categorical_crossentropy')[is_categorical]

        # looking at the model summary
        sequential_model.summary()

        # compiling the sequential model
        sequential_model.compile(loss=loss, metrics=['accuracy'], optimizer='adam')

        # training the model
        sequential_model.fit(self.model.get_x(Environment.TRAIN), self.model.get_y(Environment.TRAIN),
                             batch_size=batch_size, epochs=epochs)

        return sequential_model

    def load_model(self, name_model):
        nn_model_path = NEURAL_NETWORK_MODEL_PATH + name_model

        if not os.path.exists(nn_model_path):
            raise PathDoesNotExistException("The model needs to exists to be able to use it")

        pickels, nn_type = super(NeuralNetworkUtil, self).get_pickels_used(Structure.CategoricalNeuralNetwork,
                                                                           name_model)
        self.model.set_pickels_name(pickels)

        keras_model = keras.models.load_model(nn_model_path)

        return keras_model, nn_type

    def save_model(self, model, neural_network_type):
        model_path, model_name = self.__get_keras_model_path(neural_network_type)

        model.save(model_path + model_name)

        super(NeuralNetworkUtil, self).save_pickels_used(Structure.CategoricalNeuralNetwork, self.model.get_pickels_name(),
                                                         model_name)

        self.logger.write_info("A new categorical neural network model has been created with the name of: " + model_name
                               + "\n"
                               "In the path: " + model_path + "\n"
                               "This is the name that will be needed in the other strategies if you want to work with "
                               "this model.")

    def load_binary_zip(self):
        pass

    def record_binary_model(self, file_name, file_path):

        super(NeuralNetworkUtil, self).save_pickels_used(Structure.BinaryNeuralNetwork, self.model.get_pickels_name(),
                                                         file_name)

        self.logger.write_info("A new set of binary neural network models have been created with the name of: " +
                               file_name + "\n"
                               "In the path: " + file_path + "\n"
                               "This is the name that will be needed in the other strategies if you want to work with "
                               "these models.")

        return TMP_BINARY_CNN_MODEL_PATH

    @staticmethod
    def resize_single_image(image, nn_type):
        if nn_type == NeuralNetworkEnum.NN.value:
            resized_image = image.reshape(1, image.shape[0]*image.shape[1])
        else:
            resized_image = image.reshape(1, image.shape[0], image.shape[1], 1)

        return resized_image

    def __get_keras_model_path(self, neural_network_type):
        if not isinstance(neural_network_type, NeuralNetworkEnum):
            raise EnvironmentException("Environment used is not a valid one")

        file_name = neural_network_type.value + "_" + self.model.get_pickels_name() + "_model"

        return NEURAL_NETWORK_MODEL_PATH, file_name + ".h5"
