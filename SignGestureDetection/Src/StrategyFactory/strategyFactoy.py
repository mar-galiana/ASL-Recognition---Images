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
    """
    A factory class to get the object that will execute the strategy

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information.
    execution_strategy : string
        A value of the Strategies enumeration selecting the strategy to execute
    arguments : array
        Array of arguments entered without the execution strategy
    model : Model
        A class used to sync up all the functionalities that refer to the database
    nn_util : NeuralNetworkUtil
        A class to execute the common functionalities of a neural network structure
    decision_tree_util : DecisionTreeUtil
        A class to execute the common functionalities of a decision tree structure
    accuracy_util : AccuracyUtil
        A class to execute the common functionalities in accuracy strategies
    binary_nn_util : BinaryNeuralNetworkUtil
        A class to execute the common functionalities in the binary neural networks strategies
    storage_controller : StorageController
        A class used to remove and create the directories and files used in the execution
    strategy_switcher : dictionary
        A dictionary to select the method to execute depending on the Strategies' enumeration
    
    Methods
    -------
    get_execution_strategy()
        Get the object that will execute the strategy selected
    save_database()
        Method that will return the object that will execute the --saveDatabase strategy
    train_categorical_neural_network()
        Method that will return the object that will execute the --trainCategoricalNeuralNetwork strategy
    get_accuracy_categorical_neural_network()
        Method that will return the object that will execute the --accuracyCategoricalNeuralNetwork strategy
    train_decision_tree()
        Method that will return the object that will execute the --trainDecisionTree strategy
    get_accuracy_decision_tree()
        Method that will return the object that will execute the --accuracyDecisionTree strategy
    show_optimized_hyperparameter()
        Method that will return the object that will execute the --showOptimizedHyperparameter strategy
    train_binary_neural_network()
        Method that will return the object that will execute the --trainBinaryNeuralNetwork strategy
    get_accuracy_binary_neural_network()
        Method that will return the object that will execute the --accuracyBinaryNeuralNetwork strategy
    predict_image()
        Method that will return the object that will execute the --predict strategy
    setup()
        Method that will return the object that will execute the --setup strategy
    help()
        Method that will return the object that will execute the --help strategy
    """

    def __init__(self, logger, strategy, arguments):
        """
        Parameters
        ----------
        logger : Logger
            A class used to show the execution information.
        strategy : string
            A value of the Strategies enumeration selecting the strategy to execute
        arguments: array
            Array of arguments entered without the execution strategy
        """
        self.logger = logger
        self.execution_strategy = strategy
        self.arguments = arguments
        self.model = Model()
        self.nn_util = NeuralNetworkUtil(self.logger, self.model)
        self.decision_tree_util = DecisionTreeUtil(self.logger, self.model)
        self.accuracy_util = AccuracyUtil(self.logger, self.model)
        self.binary_nn_util = BinaryNeuralNetworkUtil(self.model)
        self.storage_controller = StorageController()

        self.strategy_switcher = self.__get_strategy_switcher()

    def get_execution_strategy(self):
        """Get the object that will execute the strategy selected

        Raises
        ------
        InputException
            If the strategy selected is not a value of the Strategies enumeration

        Returns
        -------
        IStrategy
            Returns an object implementing the IStrategy interface.
        """
        if self.execution_strategy not in self.strategy_switcher:
            raise InputException(self.execution_strategy + " is not a valid strategy")

        self.logger.write_info("Strategy selected: " + self.execution_strategy)
        strategy_method = self.strategy_switcher.get(self.execution_strategy)
        return strategy_method()

    def save_database(self):
        """Method that will return the object that will execute the --saveDatabase strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect
        
        Returns
        -------
        SaveDatabaseStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) != 3:
            raise InputException("This strategy requires four arguments to be executed")

        return SaveDatabaseStrategy(self.logger, self.model, self.arguments)

    def train_categorical_neural_network(self):
        """Method that will return the object that will execute the --trainCategoricalNeuralNetwork strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect

        Returns
        -------
        TrainCategoricalNeuralNetworkStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) < 2:
            raise InputException("This strategy requires two or more arguments to be executed")

        return TrainCategoricalNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.arguments)

    def get_accuracy_categorical_neural_network(self):
        """Method that will return the object that will execute the --accuracyCategoricalNeuralNetwork strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect

        Returns
        -------
        AccuracyCategoricalNeuralNetworkStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) != 1:
            raise InputException("This strategy requires one argument to be executed")

        return AccuracyCategoricalNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.accuracy_util,
                                                        self.arguments)

    def train_decision_tree(self):
        """Method that will return the object that will execute the --trainDecisionTree strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect

        Returns
        -------
        TrainDecisionTreeStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return TrainDecisionTreeStrategy(self.logger, self.model, self.decision_tree_util, self.arguments)

    def get_accuracy_decision_tree(self):
        """Method that will return the object that will execute the --accuracyDecisionTree strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect

        Returns
        -------
        AccuracyDecisionTreeStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) != 1:
            raise InputException("This strategy requires one argument to be executed")

        self.model.set_pickles_name(self.arguments[0])
        return AccuracyDecisionTreeStrategy(self.logger, self.model, self.decision_tree_util, self.accuracy_util,
                                            self.arguments)

    def show_optimized_hyperparameter(self):
        """Method that will return the object that will execute the --showOptimizedHyperparameter strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect

        Returns
        -------
        HyperparameterOptimizationStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return HyperparameterOptimizationStrategy(self.logger, self.model, self.nn_util, self.arguments)

    def train_binary_neural_network(self):
        """Method that will return the object that will execute the --trainBinaryNeuralNetwork strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect

        Returns
        -------
        TrainBinaryNeuralNetworkStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) < 2:
            raise InputException("This strategy requires two or more arguments to be executed")

        return TrainBinaryNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.binary_nn_util,
                                                self.storage_controller, self.arguments)

    def get_accuracy_binary_neural_network(self):
        """Method that will return the object that will execute the --accuracyBinaryNeuralNetwork strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect

        Returns
        -------
        AccuracyBinaryNeuralNetworkStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) != 1:
            raise InputException("This strategy requires only one argument to be executed")

        return AccuracyBinaryNeuralNetworkStrategy(self.logger, self.model, self.nn_util, self.accuracy_util,
                                                   self.binary_nn_util, self.storage_controller, self.arguments)

    def predict_image(self):
        """Method that will return the object that will execute the --predict strategy

        Raises
        ------
        InputException
            If the number of arguments entered is incorrect
            
        Returns
        -------
        PredictStrategy
            Returns an object implementing the IStrategy interface.
        """
        if len(self.arguments) < 1:
            raise InputException("This strategy requires one or more arguments to be executed")

        return PredictStrategy(self.logger, self.model, self.nn_util, self.decision_tree_util, self.arguments)

    def setup(self):
        """Method that will return the object that will execute the --setup strategy

        Returns
        -------
        SetupProjectStructure
            Returns an object implementing the IStrategy interface.
        """
        return SetupProjectStructure(self.logger, self.storage_controller)

    def help(self):
        """Method that will return the object that will execute the --help strategy

        Returns
        -------
        HelpStrategy
            Returns an object implementing the IStrategy interface.
        """
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
    """
    Different types of strategies to be executed
    """
    
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
