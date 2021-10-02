from enum import Enum


class Strategies(Enum):
    """
    Different types of strategies to be executed
    """

    HELP = "--help"
    SETUP = "--setup"
    PREDICT_IMAGE = "--predict"
    SAVE_DATABASE = "--saveDatabase"
    TRAIN_DECISION_TREE = "--trainDecisionTree"
    ACCURACY_DECISION_TREE = "--accuracyDecisionTree"
    TRAIN_BINARY_NEURAL_NETWORK = "--trainBinaryNeuralNetwork"
    HYPERPARAMETER_OPTIMIZATION = "--showOptimizedHyperparameter"
    ACCURACY_BINARY_NEURAL_NETWORK = "--accuracyBinaryNeuralNetwork"
    TRAIN_CATEGORICAL_NEURAL_NETWORK = "--trainCategoricalNeuralNetwork"
    ACCURACY_CATEGORICAL_NEURAL_NETWORK = "--accuracyCategoricalNeuralNetwork"
