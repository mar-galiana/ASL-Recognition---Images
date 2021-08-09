import os
import numpy as np
from skimage import io
from skimage.transform import resize
from tensorflow.python.keras.preprocessing import image
from sklearn.metrics import accuracy_score
from Model.enumerations import Environment
from StrategyFactory.iStrategy import IStrategy
from Structures.NeuralNetworks.neuralNetwork import NeuralNetwork
from Exception.inputOutputException import InputException
from Structures.NeuralNetworks.enumerations import NeuralNetworkEnum
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class AccuracyNeuralNetworkStrategy(IStrategy):

    def __init__(self, logger, model, nn_util, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.__show_arguments_entered(arguments)

        if arguments[0] not in NeuralNetworkEnum:
            raise InputException(self.arguments[0] + " is not a valid neural network")

        self.type_nn = NeuralNetworkEnum[arguments[0]]
        self.name_nn_model = arguments[1]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Neural Network type: " + arguments[0] + "\n" \
                         "\t* Neural Network model file: " + arguments[1]
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        nn, nn_model = self.__get_neural_network_model()
        self.__perform_test_data(nn, nn_model)
        self.logger.write_info("Strategy executed successfully")

    def __get_neural_network_model(self):
        if self.type_nn == NeuralNetworkEnum.CNN:
            nn = ConvolutionalNeuralNetwork(self.logger, self.model, self.nn_util)
        else:
            nn = NeuralNetwork(self.logger, self.model, self.nn_util)

        nn_model = self.nn_util.load_keras_model(self.name_nn_model)
        return nn, nn_model

    @staticmethod
    def __get_accuracy(y_pred, y_values):
        # Converting predictions to label
        prediction = list()
        for i in range(len(y_pred)):
            prediction.append(np.argmax(y_pred[i]))

        # Converting one hot encoded test label to label
        values = list()
        for i in range(len(y_values)):
            values.append(np.argmax(y_values[i]))

        accuracy = accuracy_score(prediction, values)
        return accuracy*100

    def __perform_test_data(self, nn, nn_model):

        n_classes = np.unique(self.model.get_y(Environment.TEST)).shape[0] + 1
        shape = self.model.get_x(Environment.TEST).shape

        x_test = nn.resize_data(Environment.TEST, shape)
        y_test = self.nn_util.get_categorical_vectors(Environment.TEST, n_classes)
        y_pred = nn_model.predict(x_test)

        accuracy = self.__get_accuracy(y_pred, y_test)
        self.logger.write_info("Accuracy is: " + "{:.2f}".format(accuracy) + "%")
