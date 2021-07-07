from enum import Enum


class Environment(Enum):
    TEST = "test"
    TRAIN = "train"


class Image(Enum):
    LABEL = "label"
    DATA = "data"
