import os
from Src.StrategyFactory.iStrategy import IStrategy
from Src.Exception.inputOutputException import InputException
from Src.NeuralNetworks.neuralNetwork import NeuralNetwork
from Src.NeuralNetworks.enumerations import NeuralNetworkEnum
from Src.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class TrainNeuralNetwork(IStrategy):

    def __init__(self, logger, model, nn_util, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.arguments = arguments

        self.algorithm_switcher = {
            NeuralNetworkEnum.CNN.value: ConvolutionalNeuralNetwork(self.logger, self.model, self.nn_util),
            NeuralNetworkEnum.NN.value: NeuralNetwork(self.logger, self.model, self.nn_util),
        }

    def execute(self):
        if self.arguments[0] not in self.algorithm_switcher:
            raise InputException(self.arguments[0] + " is not a valid strategy")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        algorithm_execution = self.algorithm_switcher.get(self.arguments[0])
        algorithm_execution.execute()

        self.logger.write_info("Strategy executed successfully")
