import os
import numpy as np
from skimage import io
from skimage.transform import resize
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
from Src.Model.enumerations import Environment
from Src.StrategyFactory.iStrategy import IStrategy
from Src.Exception.inputOutputException import InputException
from Src.NeuralNetworks.neuralNetwork import NeuralNetwork
from Src.NeuralNetworks.enumerations import NeuralNetworkEnum
from Src.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class AccuracyNeuralNetwork(IStrategy):

    def __init__(self, logger, model, nn_util, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.arguments = arguments

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        nn, nn_model = self.__get_neural_network_model()
        self.__perform_test_data(nn, nn_model)
        self.logger.write_info("Strategy executed successfully")

    def __get_neural_network_model(self):
        if self.arguments[0] == NeuralNetworkEnum.CNN.value:
            nn = ConvolutionalNeuralNetwork(self.logger, self.model)
            nn_model = self.nn_util.load_keras_model(NeuralNetworkEnum.CNN)
        elif self.arguments[0] == NeuralNetworkEnum.NN.value:
            nn = NeuralNetwork(self.logger, self.model)
            nn_model = self.nn_util.load_keras_model(NeuralNetworkEnum.NN)
        else:
            raise InputException(self.arguments[0] + " is not a valid neural network")

        return nn, nn_model

    @staticmethod
    def __input_prepare(shape_train):
        img_path = f"{os.getcwd()}/../Assets/Dataset/Gesture_image_data/test/0/1.jpg"
        img = io.imread(img_path, as_gray=True)
        # resize to target shape
        img = resize(img, (shape_train[1], shape_train[2]))
        # normalize
        img = img / 255
        # reshaping
        img = img.reshape(1, shape_train[1]*shape_train[2])

        return img

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
        x_test = nn.resize_data(Environment.TEST)
        y_test = self.nn_util.get_categorical_vectors(Environment.TEST, n_classes)
        y_pred = nn_model.predict(x_test)

        accuracy = self.__get_accuracy(y_pred, y_test)
        self.logger.write_info("Accuracy is: " + "{:.2f}".format(accuracy) + "%")
