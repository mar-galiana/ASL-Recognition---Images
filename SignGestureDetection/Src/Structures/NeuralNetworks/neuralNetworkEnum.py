from enum import Enum


class NeuralNetworkTypeEnum(Enum):
    CNN = "cnn"
    ANN = "ann"
    IMPROVED_CNN = "improvedcnn"


class AttributeToTuneEnum(Enum):
    BATCH_SIZE_AND_EPOCHS = "batch_and_epoch"
    OPTIMIZATION_ALGORITHMS = "optimization_algorithms"
    LEARN_RATE_AND_MOMENTUM = "learn_rate_and_momentum"
    NETWORK_WEIGHT_INITIALIZATION = "network_weight_initialization"
    NEURON_ACTIVATION_FUNCTION = "neuron_activation_function"
    DROPOUT_REGULARIZATION = "dropout_regularization"
    NUMBER_NEURONS = "number_neurons"


class ClassifierEnum(Enum):
    CLASSIFIER = 'classifier'
    SIGN = 'sign'


class LabelsRequirement(Enum):
    NUMERIC = "numeric"
    ALPHA = "alpha"
    ABC = "abc"
    ALL = "all"
