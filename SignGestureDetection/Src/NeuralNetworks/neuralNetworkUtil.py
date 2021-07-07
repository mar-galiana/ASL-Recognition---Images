import os
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from Src.Model.enumerations import Environment
from sklearn.preprocessing import LabelEncoder
from Src.NeuralNetworks.enumerations import NeuralNetworkEnum


class NeuralNetworkUtil:

    def __init__(self, model):
        self.KERAS_MODEL_BASE_PATH = f"{os.getcwd()}/../Assets/NeuralNetworkModel/"
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

    def train_model(self, sequential_model):
        # looking at the model summary
        sequential_model.summary()

        # compiling the sequential model
        sequential_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # training the model for 10 epochs
        sequential_model.fit(self.model.get_x(Environment.TRAIN),
                             self.model.get_y(Environment.TRAIN),
                             batch_size=128,
                             epochs=10,
                             validation_data=(self.model.get_x(Environment.TEST), self.model.get_y(Environment.TEST)))

        return sequential_model

    def save_keras_model(self, sequential_model, neural_network_type):
        sequential_model.save(self.__get_keras_model_path(neural_network_type))

    def load_keras_model(self, neural_network_type):
        return keras.models.load_model(self.__get_keras_model_path(neural_network_type))

    def __get_keras_model_path(self, neural_network_type):
        if not isinstance(neural_network_type, NeuralNetworkEnum):
            raise EnvironmentException("Environment used is not a valid one")

        if NeuralNetworkEnum.CNN == neural_network_type:
            file_name = "cnn_model.h5"
        else:
            file_name = "nn_model.h5"

        return self.KERAS_MODEL_BASE_PATH + file_name + ".h5"
