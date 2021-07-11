from enum import Enum
from Src.Model.model import Model
from Src.Exception.inputException import InputException
from Src.StrategyFactory.helpStrategy import HelpStrategy
from Src.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil
from Src.StrategyFactory.trainNeuralNetwork import TrainNeuralNetwork
from Src.StrategyFactory.saveDatabaseStrategy import SaveDatabaseStrategy
from Src.StrategyFactory.accuracyNeuralNetwork import AccuracyNeuralNetwork


class ExecutionFactory:

    def __init__(self, logger, strategy, arguments):
        self.logger = logger
        self.execution_strategy = strategy
        self.arguments = arguments
        self.model = Model()
        self.nn_util = NeuralNetworkUtil(self.model)

        self.strategy_switcher = {
            Strategies.HELP.value: lambda: self.help(),
            Strategies.SAVE_DATABASE.value: lambda: self.save_database(),
            Strategies.TRAIN_NEURAL_NETWORK.value: lambda: self.train_neural_network(),
            Strategies.ACCURACY_NEURAL_NETWORK.value: lambda: self.get_accuracy_neural_network()
        }

    def get_execution_strategy(self):
        if self.execution_strategy not in self.strategy_switcher:
            raise InputException(self.execution_strategy + " is not a valid strategy")

        self.logger.write_info("Strategy selected: " + self.execution_strategy)
        strategy_method = self.strategy_switcher.get(self.execution_strategy)
        return strategy_method()

    def save_database(self):
        if len(self.arguments) != 1:
            raise InputException("This strategy requires arguments to be executed")

        self.logger.write_info("Arguments entered: " + ",".join(self.arguments))
        return SaveDatabaseStrategy(self.logger, self.model, self.arguments)

    def train_neural_network(self):
        if len(self.arguments) != 1:
            raise InputException("This strategy requires arguments to be executed")

        self.logger.write_info("Arguments entered: " + ",".join(self.arguments))
        return TrainNeuralNetwork(self.logger, self.model, self.arguments)

    def get_accuracy_neural_network(self):
        if len(self.arguments) != 1:
            raise InputException("This strategy requires arguments to be executed")

        self.logger.write_info("Arguments entered: " + ",".join(self.arguments))

        return AccuracyNeuralNetwork(self.logger, self.model, self.nn_util, self.arguments)

    def help(self):
        return HelpStrategy(self.logger)


class Strategies(Enum):
    HELP = "--help"
    SAVE_DATABASE = "--saveDatabase"
    TRAIN_NEURAL_NETWORK = "--trainNeuralNetwork"
    ACCURACY_NEURAL_NETWORK = "--accuracyNeuralNetwork"
