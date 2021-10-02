from enum import Enum


class NeuralNetworkTypeEnum(Enum):
    """
    Types of neural networks.
    """

    CNN = "cnn"
    ANN = "ann"
    IMPROVED_CNN = "improvedcnn"


class AttributeToTuneEnum(Enum):
    """
    Types of attributes to be improved in the convolutional neural network model.
    """

    BATCH_SIZE_AND_EPOCHS = "batch_and_epoch"
    OPTIMIZATION_ALGORITHMS = "optimization_algorithms"
    LEARN_RATE_AND_MOMENTUM = "learn_rate_and_momentum"
    NETWORK_WEIGHT_INITIALIZATION = "network_weight_initialization"
    NEURON_ACTIVATION_FUNCTION = "neuron_activation_function"
    DROPOUT_REGULARIZATION = "dropout_regularization"
    NUMBER_NEURONS = "number_neurons"


class ClassifierEnum(Enum):
    """
    Parts of a binary classifier.
    """

    CLASSIFIER = 'classifier'
    SIGN = 'sign'


class LabelsRequirement(Enum):
    """
    Types of labels allowed to reduce the database size.
    """

    NUMERIC = "numeric"
    ALPHA = "alpha"
    ABC = "abc"
    ALL = "all"


class NeuralNetworkInformation(Enum):
    """
    Information of a neural network needed to store the training process.
    """

    Pickle = "pickle"
    Type = "type"
    Restriction = "restriction"
