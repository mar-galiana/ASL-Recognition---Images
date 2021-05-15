from enum import Enum
from Src.StrategyFactory.iStrategy import IStrategy
from Src.Exception.inputException import InputException
from Src.Algorithms.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class ExecuteAlgorithmStrategy(IStrategy):

    def __init__(self, logger, model, arguments):
        self.logger = logger
        self.model = model
        self.arguments = arguments

        self.algorithm_switcher = {
            Algorithms.CNN.value: ConvolutionalNeuralNetwork(self.logger, self.model),
        }

    def execute(self):
        if self.arguments[0] not in self.algorithm_switcher:
            raise InputException(self.arguments[0] + " is not a valid strategy")

        algorithm_execution = self.algorithm_switcher.get(self.arguments[0])
        algorithm_execution.execute()

        self.logger.write_info("Algorithm executed successfully")


class Algorithms(Enum):
    CNN = "cnn"
