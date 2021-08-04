import os
from StrategyFactory.iStrategy import IStrategy
from NeuralNetworks.neuralNetwork import NeuralNetwork
from Exception.inputOutputException import InputException
from NeuralNetworks.enumerations import NeuralNetworkEnum
from NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class TrainNeuralNetworkStrategy(IStrategy):

    def __init__(self, logger, model, nn_util, model_util, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.model_util = model_util
        self.arguments = arguments

        self.algorithm_switcher = {
            NeuralNetworkEnum.CNN.value: ConvolutionalNeuralNetwork(self.logger, self.model, self.nn_util,
                                                                    self.model_util),
            NeuralNetworkEnum.NN.value: NeuralNetwork(self.logger, self.model, self.nn_util, self.model_util),
        }

    def execute(self):
        if self.arguments[1] not in self.algorithm_switcher:
            raise InputException(self.arguments[1] + " is not a valid strategy")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        algorithm_execution = self.algorithm_switcher.get(self.arguments[1])
        algorithm_execution.execute()

        self.logger.write_info("Strategy executed successfully")
