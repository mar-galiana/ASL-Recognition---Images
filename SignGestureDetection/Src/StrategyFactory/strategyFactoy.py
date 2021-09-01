from enum import Enum
from Model.model import Model
from StrategyFactory.accuracyUtil import AccuracyUtil
from StrategyFactory.helpStrategy import HelpStrategy
from Storage.storageController import StorageController
from Exception.inputOutputException import InputException
from StrategyFactory.predictStrategy import PredictStrategy
from Structures.DecisionTree.decisionTreeUtil import DecisionTreeUtil
from StrategyFactory.saveDatabaseStrategy import SaveDatabaseStrategy
from Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil
from StrategyFactory.trainDecisionTreeStrategy import TrainDecisionTreeStrategy
from StrategyFactory.accuracyDecisionTreeStrategy import AccuracyDecisionTreeStrategy
from StrategyFactory.accuracyNeuralNetworkStrategy import AccuracyNeuralNetworkStrategy
from StrategyFactory.hyperparameterOptimizationStrategy import HyperparameterOptimizationStrategy
from Src.StrategyFactory.trainBinaryNeuralNetworkStrategy import TrainBinaryNeuralNetworkStrategy
from StrategyFactory.trainCategoricalNeuralNetworkStrategy import TrainCategoricalNeuralNetworkStrategy


class ExecutionFactory:

    def __init__(self, logger, strategy, arguments):
        self.logger = logger
        self.execution_strategy = strategy
        self.arguments = arguments
        self.model = Model()
        self.nn_util = NeuralNetworkUtil(self.logger, self.model)
        self.decision_tree_util = DecisionTreeUtil(self.logger, self.model)
        self.accuracy_util = AccuracyUtil(self.model, self.logger)
        self.storage_controller = StorageController()

        self.strategy_switcher = {
            Strategies.HELP.value: lambda: self.help(),
            Strategies.SAVE_DATABASE.value: lambda: self.save_database(),
            Strategies.TRAIN_CATEGORICAL_NEURAL_NETWORK.value: lambda: self.train_categorical_neural_network(),
            Strategies.ACCURACY_NEURAL_NETWORK.value: lambda: self.get_accuracy_neural_network(),
            Strategies.DECISION_TREE.value: lambda: self.train_decision_tree(),
            Strategies.ACCURACY_DECISION_TREE.value: lambda: self.get_accuracy_decision_tree(),
            Strategies.HYPERPARAMETER_OPTIMIZATION.value: lambda: self.show_optimized_hyperparameter(),
            Strategies.PREDICT_IMAGE.value: lambda: self.predict_image(),
            Strategies.TRAIN_BINARY_NEURAL_NETWORK.value: lambda: self.train_binary_neural_network(),
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

    def train_categorical_neural_network(self):
        if len(self.arguments) < 2:
            raise InputException("This strategy requires two or more arguments to be executed")

        return TrainCategoricalNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.arguments)

    def get_accuracy_neural_network(self):
        if len(self.arguments) != 1:
            raise InputException("This strategy requires one argument to be executed")

        return AccuracyNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.accuracy_util, self.arguments)

    def train_decision_tree(self):
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return TrainDecisionTreeStrategy(self.logger, self.model, self.decision_tree_util, self.arguments)

    def get_accuracy_decision_tree(self):
        if len(self.arguments) != 1:
            raise InputException("This strategy requires one argument to be executed")

        self.model.set_pickels_name(self.arguments[0])
        return AccuracyDecisionTreeStrategy(self.logger, self.model, self.decision_tree_util, self.accuracy_util,
                                            self.arguments)

    def show_optimized_hyperparameter(self):
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return HyperparameterOptimizationStrategy(self.logger, self.model, self.nn_util, self.arguments)

    def train_binary_neural_network(self):
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return TrainBinaryNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.storage_controller,
                                                self.arguments)

    def predict_image(self):
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return PredictStrategy(self.logger, self.model, self.nn_util, self.decision_tree_util, self.arguments)

    def help(self):
        return HelpStrategy(self.logger)


class Strategies(Enum):
    HELP = "--help"
    SAVE_DATABASE = "--saveDatabase"
    TRAIN_CATEGORICAL_NEURAL_NETWORK = "--trainCategoricalNeuralNetwork"
    ACCURACY_NEURAL_NETWORK = "--accuracyNeuralNetwork"
    DECISION_TREE = "--trainDecisionTree"
    ACCURACY_DECISION_TREE = "--accuracyDecisionTree"
    HYPERPARAMETER_OPTIMIZATION = "--showOptimizedHyperparameter"
    PREDICT_IMAGE = "--predict"
    TRAIN_BINARY_NEURAL_NETWORK = "--trainBinaryNeuralNetwork"
