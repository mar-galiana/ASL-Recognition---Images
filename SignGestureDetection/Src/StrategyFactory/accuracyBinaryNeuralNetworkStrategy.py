import os
import numpy as np
from skimage import io
from skimage.transform import resize
from Model.modelEnum import Environment
from StrategyFactory.iStrategy import IStrategy
from tensorflow.python.keras.preprocessing import image
from Exception.inputOutputException import InputException
from Structures.NeuralNetworks.neuralNetwork import NeuralNetwork
from Structures.NeuralNetworks.neuralNetworkEnum import ClassifierEnum
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Constraints.path import TMP_BINARY_NEURAL_NETWORK_MODEL_PATH, BINARY_NEURAL_NETWORK_MODEL_PATH


class AccuracyBinaryNeuralNetworkStrategy(IStrategy):

    def __init__(self, logger, model, nn_util, accuracy_util, storage_controller, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.accuracy_util = accuracy_util
        self.storage_controller = storage_controller

        self.__show_arguments_entered(arguments)

        self.name_nn_model = arguments[0]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Neural Network model file: " + arguments[0]
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.__prepare_model()
        self.__remove_not_wanted_labels()
        classifiers = self.__get_classifiers_models()
        self.perform_test_data(classifiers)
        self.__remove_temporal_files()
        self.logger.write_info("Strategy executed successfully")

    def perform_test_data(self, classifiers):

        shape = self.model.get_x(Environment.TEST).shape

        x_test = self.__resize_data(Environment.TEST, shape)
        y_test = self.model.get_sign_values(self.model.get_y(Environment.TEST))

        for classifier_dict in classifiers:
            classifier = classifier_dict[ClassifierEnum.CLASSIFIER.value]
            numeric_sign = self.model.get_sign_value(classifier_dict[ClassifierEnum.SIGN.value])
            y_pred = classifier.predict(x_test)
            accuracy = self.__get_accuracy(y_test, y_pred, numeric_sign)
            self.logger.write_info("Accuracy " + classifier_dict[ClassifierEnum.SIGN.value] + " classifier is: " +
                                   "{:.2f}".format(accuracy) + "%")

    def __prepare_model(self):
        pickels = self.nn_util.get_pickels_used_in_binary_zip(self.name_nn_model)
        self.model.set_pickels_name(pickels)

    def __get_classifiers_models(self):
        source_path = BINARY_NEURAL_NETWORK_MODEL_PATH + self.name_nn_model
        destination_path = TMP_BINARY_NEURAL_NETWORK_MODEL_PATH + self.name_nn_model[:-len(".zip")]
        model_files_path = self.storage_controller.extract_compressed_files(source_path, destination_path)

        classifiers = []
        for model_path in model_files_path:
            # Get classifier symbol
            sign = model_path[-len(".h5") - 1]

            classifier = self.nn_util.read_model(model_path)
            classifiers.append({
                ClassifierEnum.CLASSIFIER.value: classifier,
                ClassifierEnum.SIGN.value: sign
            })

        return classifiers

    def __resize_data(self, environment, shape):
        x_data = self.model.get_x(environment).reshape(shape[0], shape[1], shape[2], 1)
        return x_data

    def __remove_temporal_files(self):
        directory_path = TMP_BINARY_NEURAL_NETWORK_MODEL_PATH + self.name_nn_model[:-len(".zip")]
        self.storage_controller.remove_files_from_folder(directory_path)

    def __remove_not_wanted_labels(self):
        y_test = self.model.get_y(Environment.TEST)
        x_test = self.model.get_x(Environment.TEST)
        indexes = [i for i, label in enumerate(y_test) if label != 'A' and label != 'B' and label != 'C']
        y_test = np.delete(y_test, indexes)
        x_test = np.delete(x_test, indexes, axis=0)

        self.model.set_y(Environment.TEST, y_test)
        self.model.set_x(Environment.TEST, x_test)

    @staticmethod
    def __get_accuracy(y_test, y_pred, sign):
        correct = 0
        for test, pred in zip(y_test, y_pred):
            if (sign == test and pred[0] > 0.5) or (sign != test and pred[0] < 0.5):
                correct += 1

        return (correct / len(y_test)) * 100
