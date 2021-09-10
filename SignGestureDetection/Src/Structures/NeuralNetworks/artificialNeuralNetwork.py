import numpy as np
from Model.modelEnum import Environment
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from Structures.NeuralNetworks.iNeuralNetwork import INeuralNetwork
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil


class ArtificialNeuralNetwork(INeuralNetwork):

    def __init__(self, logger, model, nn_util):
        self.model = model
        self.logger = logger
        self.nn_util = nn_util

    def resize_data(self, environment, shape):
        if len(shape) > 3:
            x_data = self.model.get_x(environment).reshape(shape[0], shape[1]*shape[2], shape[3])
        else:
            x_data = self.model.get_x(environment).reshape(shape[0], shape[1]*shape[2])
        return x_data

    def train_neural_network(self):
        shape_train = self.model.get_x(Environment.TRAIN).shape
        shape_test = self.model.get_x(Environment.TEST).shape
        n_classes = self.__prepare_images(shape_test, shape_train)
        sequential_model = self.__build_sequential_model(n_classes, shape_train)
        self.nn_util.save_model(sequential_model, NeuralNetworkTypeEnumNN)

    def __prepare_images(self, shape_test, shape_train):
        # Flattening the images from the 150x150 pixels to 1D 787 pixels
        x_train = self.resize_data(Environment.TRAIN, shape_train)
        x_test = self.resize_data(Environment.TEST, shape_test)

        self.model.set_x(Environment.TRAIN, x_train)
        self.model.set_x(Environment.TEST, x_test)

        n_classes = self.model.convert_to_one_hot_data()
        return n_classes

    def __build_sequential_model(self, n_classes, shape_train):
        # building a linear stack of layers with the sequential model
        sequential_model = Sequential()
        # hidden layer
        if len(shape_train) > 3:
            sequential_model.add(Dense(100, input_shape=(shape_train[1]*shape_train[2], shape_train[3]),
                                       activation='relu'))
        else:
            sequential_model.add(Dense(100, input_shape=(shape_train[1]*shape_train[2],), activation='relu'))

        # output layer
        sequential_model.add(Dense(n_classes, activation='softmax'))
        sequential_model = self.nn_util.train_model(sequential_model)
        return sequential_model
