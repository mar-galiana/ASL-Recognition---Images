from enum import Enum


class Environment(Enum):
    TEST = "test"
    TRAIN = "train"


class Image(Enum):
    LABEL = "label"
    DATA = "data"
    DESCRIPTION = "description"


class Dataset(Enum):
    GESTURE_IMAGE_DATA = "GestureImageData"
    GESTURE_IMAGE_PRE_PROCESSED_DATA = "GestureImagePreProcessedData"
    ASL_ALPHABET = "AslAlphabet"
