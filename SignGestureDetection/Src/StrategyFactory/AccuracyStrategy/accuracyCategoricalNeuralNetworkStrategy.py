import os
from StrategyFactory.iStrategy import IStrategy
from Structures.iUtilStructure import Structure
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Structures.NeuralNetworks.artificialNeuralNetwork import ArtificialNeuralNetwork
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

        nn_model, nn_type = self.nn_util.load_model(self.name_nn_model)

        self.accuracy_util.perform_test_data(Structure.CategoricalNeuralNetwork, nn_model, nn_type=nn_type)
        self.logger.write_info("Strategy executed successfully")
