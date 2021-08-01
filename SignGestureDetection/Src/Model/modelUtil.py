import numpy as np
from tensorflow import keras
from tensorflow.python.keras.utils import np_utils
from Model.enumerations import Environment
from sklearn.preprocessing import LabelEncoder
from Exception.modelException import EnvironmentException


class ModelUtil:
    def __init__(self, model):
        self.model = model

    def get_categorical_vectors(self, environment, n_classes):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        label_encoder = LabelEncoder()

        vectors = label_encoder.fit_transform(self.model.get_y(environment))
        y_data = np_utils.to_categorical(vectors, num_classes=n_classes)
        return y_data

    def convert_to_one_hot_data(self):
        x_train = self.model.get_x(Environment.TRAIN).astype('float32')
        x_test = self.model.get_x(Environment.TEST).astype('float32')

        # normalizing the data to help with the training
        x_train /= 255
        x_test /= 255

        # one-hot encoding using keras' numpy-related utilities
        n_classes = np.unique(self.model.get_y(Environment.TRAIN)).shape[0] + 1
        y_train = self.get_categorical_vectors(Environment.TRAIN, n_classes)
        y_test = self.get_categorical_vectors(Environment.TEST, n_classes)

        self.model.set_y(Environment.TRAIN, y_train)
        self.model.set_y(Environment.TEST, y_test)

        return n_classes
