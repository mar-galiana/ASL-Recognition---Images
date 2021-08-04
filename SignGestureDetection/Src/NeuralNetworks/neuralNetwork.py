import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from Model.enumerations import Environment
from NeuralNetworks.iNeuralNetwork import INeuralNetwork
from NeuralNetworks.enumerations import NeuralNetworkEnum
from NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil


class NeuralNetwork(INeuralNetwork):

    def __init__(self, logger, model, nn_util, model_util):
        self.model = model
        self.logger = logger
        self.nn_util = nn_util
        self.model_util = model_util

    def resize_data(self, environment, shape):
        if len(shape) > 3:
            x_data = self.model.get_x(environment).reshape(shape[0], shape[1]*shape[2], shape[3])
        else:
            x_data = self.model.get_x(environment).reshape(shape[0], shape[1]*shape[2])
        return x_data

    def execute(self):
        self.logger.write_info("Execute")
        shape_train = self.model.get_x(Environment.TRAIN).shape
        shape_test = self.model.get_x(Environment.TEST).shape
        self.logger.write_info("Shapes")
        n_classes = self.__prepare_images(shape_test, shape_train)
        sequential_model = self.__build_sequential_model(n_classes, shape_train)
        self.logger.write_info("pre save")
        self.nn_util.save_keras_model(sequential_model, NeuralNetworkEnum.NN)
        self.logger.write_info("post save")

    def __prepare_images(self, shape_test, shape_train):
        # Flattening the images from the 150x150 pixels to 1D 787 pixels
        x_train = self.resize_data(Environment.TRAIN, shape_train)
        self.logger.write_info("Resize train")
        x_test = self.resize_data(Environment.TEST, shape_test)
        self.logger.write_info("Resize test")

        self.model.set_x(Environment.TRAIN, x_train)
        self.model.set_x(Environment.TEST, x_test)

        n_classes = self.model_util.convert_to_one_hot_data()
        self.logger.write_info("convert_to_one_hot_data")
        return n_classes

    def __build_sequential_model(self, n_classes, shape_train):
        # building a linear stack of layers with the sequential model
        sequential_model = Sequential()
        self.logger.write_info("Sequential")
        # hidden layer
        if len(shape) > 3:
            sequential_model.add(Dense(100, input_shape=(shape_train[1]*shape_train[2], shape_train[3]),
                                       activation='relu'))
        else:
            sequential_model.add(Dense(100, input_shape=(shape_train[1]*shape_train[2],), activation='relu'))

        self.logger.write_info("Add hidden layer")
        # output layer
        sequential_model.add(Dense(n_classes, activation='softmax'))
        self.logger.write_info("pre train model")
        sequential_model = self.nn_util.train_model(sequential_model)
        self.logger.write_info("post train model")
        return sequential_model
