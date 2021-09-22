from enum import Enum


class Environment(Enum):
    """
    Types of environments in a dataset
    """

    TEST = "test"
    TRAIN = "train"


class Image(Enum):
    """
    Types of data in a sample
    """

    LABEL = "label"
    DATA = "data"
    DESCRIPTION = "description"


class Dataset(Enum):
    """
    Types of datasets
    """

    GESTURE_IMAGE_DATA = "GestureImageData"
    GESTURE_IMAGE_PRE_PROCESSED_DATA = "GestureImagePreProcessedData"
    ASL_ALPHABET = "AslAlphabet"
