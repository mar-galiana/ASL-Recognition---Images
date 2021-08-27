from enum import Enum


class NeuralNetworkEnum(Enum):
    CNN = "cnn"
    NN = "nn"
    IMPROVED_CNN = "improved_cnn"


class AttributeToTune(Enum):
    BATCH_SIZE_AND_EPOCHS = "batch_and_epoch"
    OPTIMIZATION_ALGORITHMS = "optimization_algorithms"

