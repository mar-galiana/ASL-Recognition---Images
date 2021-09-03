"""
One major advantage of using CNNs over NNs is that you do not need to flatten the input images to 1D as they are capable
of working with image data in 2D.
"""
import numpy as np
from Constraints.hyperparameters import *
from Model.modelEnum import Environment
from tensorflow.keras.constraints import max_norm
from tensorflow.python.keras.models import Sequential
from Structures.NeuralNetworks.iNeuralNetwork import INeuralNetwork
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Exception.parametersException import IncorrectNumberOfParameters
from Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


class ConvolutionalNeuralNetwork(INeuralNetwork):

    def __init__(self, logger, model, nn_util, improved_nn=False):

        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.improved_nn = improved_nn

    def resize_data(self, environment, shape):
        x_data = self.model.get_x(environment).reshape(shape[0], shape[1], shape[2], 1)
        return x_data

    def train_neural_network(self):
        if self.improved_nn is None:
            raise IncorrectNumberOfParameters("Convolutional Neural Network needs the improved_nn parameter if it has "
                                              "to be trained")

        shape_train = self.model.get_x(Environment.TRAIN).shape
        shape_test = self.model.get_x(Environment.TEST).shape
        n_classes = self.prepare_images(shape_train, shape_test)
        sequential_model = self.build_sequential_model(n_classes, shape_train)

        nn_type = (NeuralNetworkTypeEnumCNN, NeuralNetworkTypeEnumIMPROVED_CNN)[self.improved_nn]
        self.nn_util.save_model(sequential_model, nn_type)

    def prepare_images(self, shape_train, shape_test):

        # Flattening the images from the 150x150 pixels to 1D 787 pixels
        x_train = self.resize_data(Environment.TRAIN, shape_train)
        x_test = self.resize_data(Environment.TEST, shape_test)

        self.model.set_x(Environment.TRAIN, x_train)
        self.model.set_x(Environment.TEST, x_test)

        n_classes = self.model.convert_to_one_hot_data()

        return n_classes

    def build_sequential_model(self, n_classes, shape, is_categorical=True):

        if self.improved_nn:
            seq_model = self.__get_improved_sequential_model(n_classes, shape, is_categorical)
            seq_model = self.nn_util.train_model(seq_model, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                                 is_categorical=is_categorical)

        else:
            seq_model = self.__get_not_improved_sequential_model(n_classes, shape)
            seq_model = self.nn_util.train_model(seq_model)

        return seq_model

    @staticmethod
    def __get_improved_sequential_model(n_classes, shape, is_categorical):
        activation = ('sigmoid', 'softmax')[is_categorical]

        sequential_model = Sequential()
        sequential_model.add(Conv2D(NEURONS_CONV_LAYER, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                    activation=ACTIVATION, input_shape=(shape[1], shape[2], 1),
                                    kernel_initializer=INIT_MODE, kernel_constraint=max_norm(WEIGHT_CONSTRAINT)))
        sequential_model.add(MaxPool2D(pool_size=(1, 1)))
        sequential_model.add(Flatten())
        sequential_model.add(Dense(NEURONS_DENSE_LAYER, kernel_initializer=INIT_MODE, activation=ACTIVATION))
        sequential_model.add(Dense(n_classes, activation=activation))

        return sequential_model

    @staticmethod
    def __get_not_improved_sequential_model(n_classes, shape):
        # building a linear stack of layers with the sequential model
        sequential_model = Sequential()

        # convolutional layer
        sequential_model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                                    input_shape=(shape[1], shape[2], 1)))

        # pooling Layers: Prevent overfitting
        sequential_model.add(MaxPool2D(pool_size=(1, 1)))

        # flatten output of conv
        sequential_model.add(Flatten())

        # hidden layer
        sequential_model.add(Dense(100, activation='relu'))

        # output layer ->   the softmax activation function selects the neuron with the highest probability as its
        #                   output, voting that the image belongs to that class
        sequential_model.add(Dense(n_classes, activation='softmax'))

        return sequential_model
