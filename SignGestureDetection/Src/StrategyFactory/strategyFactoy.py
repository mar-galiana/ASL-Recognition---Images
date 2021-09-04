from enum import Enum
from Model.model import Model
from StrategyFactory.helpStrategy import HelpStrategy
from Storage.storageController import StorageController
from Exception.inputOutputException import InputException
from StrategyFactory.predictStrategy import PredictStrategy
from StrategyFactory.Utils.accuracyUtil import AccuracyUtil
from StrategyFactory.saveDatabaseStrategy import SaveDatabaseStrategy
from Structures.DecisionTree.decisionTreeUtil import DecisionTreeUtil
from StrategyFactory.setupProjectStrategy import SetupProjectStructure
from Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil
from StrategyFactory.Utils.binaryNeuralNetworkUtil import BinaryNeuralNetworkUtil
from StrategyFactory.TrainStrategy.trainDecisionTreeStrategy import TrainDecisionTreeStrategy
from StrategyFactory.hyperparameterOptimizationStrategy import HyperparameterOptimizationStrategy
from StrategyFactory.AccuracyStrategy.accuracyDecisionTreeStrategy import AccuracyDecisionTreeStrategy
from StrategyFactory.TrainStrategy.trainBinaryNeuralNetworkStrategy import TrainBinaryNeuralNetworkStrategy
from StrategyFactory.AccuracyStrategy.accuracyBinaryNeuralNetworkStrategy import AccuracyBinaryNeuralNetworkStrategy
from StrategyFactory.TrainStrategy.trainCategoricalNeuralNetworkStrategy import TrainCategoricalNeuralNetworkStrategy
from StrategyFactory.AccuracyStrategy.accuracyCategoricalNeuralNetworkStrategy import AccuracyCategoricalNeuralNetworkStrategy


class ExecutionFactory:

    def __init__(self, logger, strategy, arguments):
        self.logger = logger
        self.execution_strategy = strategy
        self.arguments = arguments
        self.model = Model()
        self.nn_util = NeuralNetworkUtil(self.logger, self.model)
        self.decision_tree_util = DecisionTreeUtil(self.logger, self.model)
        self.accuracy_util = AccuracyUtil(self.model, self.logger)
        self.binary_nn_util = BinaryNeuralNetworkUtil(self.model)
        self.storage_controller = StorageController()

        self.strategy_switcher = self.__get_strategy_switcher()

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

    def get_accuracy_categorical_neural_network(self):
        if len(self.arguments) != 1:
            raise InputException("This strategy requires one argument to be executed")

        return AccuracyCategoricalNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.accuracy_util,
                                                        self.arguments)

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
        if len(self.arguments) < 2:
            raise InputException("This strategy requires two or more arguments to be executed")

        return TrainBinaryNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.binary_nn_util,
                                                self.storage_controller, self.arguments)

    def get_accuracy_binary_neural_network(self):
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return AccuracyBinaryNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.accuracy_util,
                                                   self.binary_nn_util, self.storage_controller, self.arguments)

    def predict_image(self):
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return PredictStrategy(self.logger, self.model, self.nn_util, self.decision_tree_util, self.arguments)

    def setup(self):
        return SetupProjectStructure(self.logger, self.storage_controller)

    def help(self):
        return HelpStrategy(self.logger)

    def __get_strategy_switcher(self):
        return {
            Strategies.HELP.value: lambda: self.help(),
            Strategies.SETUP.value: lambda: self.setup(),
            Strategies.SAVE_DATABASE.value: lambda: self.save_database(),
            Strategies.PREDICT_IMAGE.value: lambda: self.predict_image(),
            Strategies.DECISION_TREE.value: lambda: self.train_decision_tree(),
            Strategies.ACCURACY_DECISION_TREE.value: lambda: self.get_accuracy_decision_tree(),
            Strategies.TRAIN_BINARY_NEURAL_NETWORK.value: lambda: self.train_binary_neural_network(),
            Strategies.HYPERPARAMETER_OPTIMIZATION.value: lambda: self.show_optimized_hyperparameter(),
            Strategies.ACCURACY_BINARY_NEURAL_NETWORK.value: lambda: self.get_accuracy_binary_neural_network(),
            Strategies.TRAIN_CATEGORICAL_NEURAL_NETWORK.value: lambda: self.train_categorical_neural_network(),
            Strategies.ACCURACY_CATEGORICAL_NEURAL_NETWORK.value: lambda: self.get_accuracy_categorical_neural_network()
        }


class Strategies(Enum):
    HELP = "--help"
    SETUP = "--setup"
    PREDICT_IMAGE = "--predict"
    SAVE_DATABASE = "--saveDatabase"
    DECISION_TREE = "--trainDecisionTree"
    ACCURACY_DECISION_TREE = "--accuracyDecisionTree"
    TRAIN_BINARY_NEURAL_NETWORK = "--trainBinaryNeuralNetwork"
    HYPERPARAMETER_OPTIMIZATION = "--showOptimizedHyperparameter"
    ACCURACY_BINARY_NEURAL_NETWORK = "--accuracyBinaryNeuralNetwork"
    TRAIN_CATEGORICAL_NEURAL_NETWORK = "--trainCategoricalNeuralNetwork"
    ACCURACY_CATEGORICAL_NEURAL_NETWORK = "--accuracyCategoricalNeuralNetwork"
