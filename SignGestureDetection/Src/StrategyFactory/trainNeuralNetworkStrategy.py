import os
from StrategyFactory.iStrategy import IStrategy
from Structures.NeuralNetworks.neuralNetwork import NeuralNetwork
from Exception.inputOutputException import InputException
from Structures.NeuralNetworks.enumerations import NeuralNetworkEnum
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class TrainNeuralNetworkStrategy(IStrategy):

    def __init__(self, logger, model, nn_util, model_util, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.model_util = model_util
        self.__show_arguments_entered(arguments)

        self.nn_type = arguments[0]
        self.pickels = arguments[1:]

        self.algorithm_switcher = {
            NeuralNetworkEnum.CNN.value: ConvolutionalNeuralNetwork(self.logger, self.model, self.nn_util,
                                                                    self.model_util),
            NeuralNetworkEnum.NN.value: NeuralNetwork(self.logger, self.model, self.nn_util, self.model_util),
        }

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Neural Network type: " + arguments[0] + "\n" \
                         "\t* Pickels selected: " + ", ".join(arguments[1:])
        self.logger.write_info(info_arguments)

    def execute(self):
        if self.nn_type not in self.algorithm_switcher:
            raise InputException(self.nn_type + " is not a valid strategy")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.model.set_pickels_name(self.pickels)

        algorithm_execution = self.algorithm_switcher.get(self.nn_type)
        algorithm_execution.train_neural_network()

        self.logger.write_info("Strategy executed successfully")
