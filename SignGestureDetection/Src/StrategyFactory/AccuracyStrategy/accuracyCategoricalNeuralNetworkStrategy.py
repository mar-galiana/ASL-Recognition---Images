import os
import numpy as np
from skimage import io
from skimage.transform import resize
from Model.modelEnum import Environment
from StrategyFactory.iStrategy import IStrategy
from tensorflow.python.keras.preprocessing import image
from Exception.inputOutputException import InputException
from Structures.NeuralNetworks.neuralNetwork import NeuralNetwork
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class AccuracyCategoricalNeuralNetworkStrategy(IStrategy):

    def __init__(self, logger, model, nn_util, accuracy_util, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.accuracy_util = accuracy_util

        self.__show_arguments_entered(arguments)

        self.name_nn_model = arguments[0]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Neural Network model file: " + arguments[0]
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        nn, nn_model, nn_type = self.__get_neural_network_model()
        self.accuracy_util.perform_test_data(nn, nn_model, nn_type=nn_type)
        self.logger.write_info("Strategy executed successfully")

    def __get_neural_network_model(self):
        nn_model, nn_type = self.nn_util.load_model(self.name_nn_model)

        if nn_type == NeuralNetworkTypeEnumNN:
            nn = NeuralNetwork(self.logger, self.model, self.nn_util)
        else:
            nn = ConvolutionalNeuralNetwork(self.logger, self.model, self.nn_util)

        return nn, nn_model, nn_type
