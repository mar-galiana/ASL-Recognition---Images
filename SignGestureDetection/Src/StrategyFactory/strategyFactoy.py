from enum import Enum
from Model.model import Model
from Model.modelUtil import ModelUtil
from StrategyFactory.helpStrategy import HelpStrategy
from Exception.inputOutputException import InputException
from Structures.DecisionTree.decisionTreeUtil import DecisionTreeUtil
from Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil
from StrategyFactory.decisionTreeStrategy import DecisionTreeStrategy
from StrategyFactory.saveDatabaseStrategy import SaveDatabaseStrategy
from StrategyFactory.trainNeuralNetworkStrategy import TrainNeuralNetworkStrategy
from StrategyFactory.accuracyDecisionTreeStrategy import AccuracyDecisionTreeStrategy
from StrategyFactory.accuracyNeuralNetworkStrategy import AccuracyNeuralNetworkStrategy


class ExecutionFactory:

    def __init__(self, logger, strategy, arguments):
        self.logger = logger
        self.execution_strategy = strategy
        self.arguments = arguments
        self.model = Model()
        self.model_util = ModelUtil(self.model)
        self.nn_util = NeuralNetworkUtil(self.logger, self.model)
        self.decision_tree_util = DecisionTreeUtil(self.logger, self.model)

        self.strategy_switcher = {
            Strategies.HELP.value: lambda: self.help(),
            Strategies.SAVE_DATABASE.value: lambda: self.save_database(),
            Strategies.TRAIN_NEURAL_NETWORK.value: lambda: self.train_neural_network(),
            Strategies.ACCURACY_NEURAL_NETWORK.value: lambda: self.get_accuracy_neural_network(),
            Strategies.DECISION_TREE.value: lambda: self.decision_tree(),
            Strategies.ACCURACY_DECISION_TREE.value: lambda: self.get_accuracy_decision_tree()
        }

    def get_execution_strategy(self):
        if self.execution_strategy not in self.strategy_switcher:
            raise InputException(self.execution_strategy + " is not a valid strategy")

        self.logger.write_info("Strategy selected: " + self.execution_strategy)
        strategy_method = self.strategy_switcher.get(self.execution_strategy)
        return strategy_method()

    def save_database(self):
        if len(self.arguments) != 4:
            raise InputException("This strategy requires four arguments to be executed")

        return SaveDatabaseStrategy(self.logger, self.model, self.arguments)

    def train_neural_network(self):
        if len(self.arguments) < 2:
            raise InputException("This strategy requires two or more arguments to be executed")

        return TrainNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.model_util, self.arguments)

    def get_accuracy_neural_network(self):
        if len(self.arguments) != 2:
            raise InputException("This strategy requires two arguments to be executed")

        return AccuracyNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.model_util, self.arguments)

    def decision_tree(self):
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return DecisionTreeStrategy(self.logger, self.model, self.decision_tree_util, self.model_util, self.arguments)

    def get_accuracy_decision_tree(self):
        if len(self.arguments) != 1:
            raise InputException("This strategy requires one argument to be executed")

        self.model.set_pickels_name(self.arguments[0])
        return AccuracyDecisionTreeStrategy(self.logger, self.model, self.decision_tree_util, self.arguments)

    def help(self):
        return HelpStrategy(self.logger)


class Strategies(Enum):
    HELP = "--help"
    SAVE_DATABASE = "--saveDatabase"
    TRAIN_NEURAL_NETWORK = "--trainNeuralNetwork"
    ACCURACY_NEURAL_NETWORK = "--accuracyNeuralNetwork"
    DECISION_TREE = "--decisionTree"
    ACCURACY_DECISION_TREE = "--accuracyDecisionTree"
