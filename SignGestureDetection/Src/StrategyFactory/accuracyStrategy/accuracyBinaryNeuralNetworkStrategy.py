import os
import numpy as np
from skimage import io
from skimage.transform import resize
from Model.modelEnum import Environment
from StrategyFactory.iStrategy import IStrategy
from tensorflow.python.keras.preprocessing import image
from Exception.inputOutputException import InputException
from Exception.modelException import DifferentPickelsException
from Structures.NeuralNetworks.neuralNetwork import NeuralNetwork
from Structures.NeuralNetworks.neuralNetworkEnum import ClassifierEnum
from Structures.NeuralNetworks.neuralNetworkEnum import LabelsRequirement
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Constraints.path import TMP_BINARY_NEURAL_NETWORK_MODEL_PATH, BINARY_NEURAL_NETWORK_MODEL_PATH


class AccuracyBinaryNeuralNetworkStrategy(IStrategy):

    ACCURACY_PERCENTAGE = 0.75

    def __init__(self, logger, model, nn_util, accuracy_util, storage_controller, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.accuracy_util = accuracy_util
        self.storage_controller = storage_controller

        self.__show_arguments_entered(arguments)

        if arguments[0] not in LabelsRequirement._value2member_map_:
            raise InputException(self.execution_strategy + " is not a valid sign requirement")

        self.labels_requirement = LabelsRequirement(arguments[0])
        self.models_names = arguments[1:]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Signs to train: " + arguments[0] + "\n" \
                         "\t* Neural Network model file: " + ", ".join(arguments)
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        classifiers = self.__get_classifiers()

        self.__remove_not_wanted_labels()

        self.perform_test_data(classifiers)

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
            self.logger.write_info("The accuracy of the binary neural network of the sign '" +
                                   classifier_dict[ClassifierEnum.SIGN.value] + "' is: {:.2f}".format(accuracy) + "%")

    def __get_model_pickels(self, model_name):
        pickels = self.nn_util.get_pickels_used_in_binary_zip(model_name)

        return pickels

    def __get_classifiers(self):
        classifiers = []
        global_pickels = None

        for model_name in self.models_names:
            model_pickels = self.__get_model_pickels(model_name)
            classifiers += self.__get_classifiers_from_specific_models(model_name)
            self.__remove_temporal_files(model_name)

            if global_pickels is None:
                global_pickels = model_pickels

            elif set(model_pickels) != set(global_pickels):
                raise DifferentPickelsException("Selected models use different pickles, be sure to select models "
                                                "trained with the same pickles.")

        self.model.set_pickels_name(global_pickels)

        return classifiers

    def __get_classifiers_from_specific_models(self, model_name):
        source_path = BINARY_NEURAL_NETWORK_MODEL_PATH + model_name
        destination_path = TMP_BINARY_NEURAL_NETWORK_MODEL_PATH + model_name[:-len(".zip")]
        model_files_path = self.storage_controller.extract_compressed_files(source_path, destination_path)

        classifiers = []
        for model_path in model_files_path:
            # Get classifier symbol
            sign_array = model_path[:-len(".h5")].split('_')
            sign = sign_array[len(sign_array)-1]

            classifier = self.nn_util.read_model(model_path)
            classifiers.append({
                ClassifierEnum.CLASSIFIER.value: classifier,
                ClassifierEnum.SIGN.value: sign
            })

        return classifiers

    def __resize_data(self, environment, shape):
        x_data = self.model.get_x(environment).reshape(shape[0], shape[1], shape[2], 1)
        return x_data

    def __remove_temporal_files(self, model_name):
        directory_path = TMP_BINARY_NEURAL_NETWORK_MODEL_PATH + model_name[:-len(".zip")]
        self.storage_controller.remove_files_from_folder(directory_path)

    def __remove_not_wanted_labels(self):

        y_train = self.model.get_y(Environment.TEST)
        x_train = self.model.get_x(Environment.TEST)

        indexes = [i for i, label in enumerate(y_train) if self.__is_required(label)]

        y_train = np.delete(y_train, indexes)
        x_train = np.delete(x_train, indexes, axis=0)

        self.model.set_y(Environment.TEST, y_train)
        self.model.set_x(Environment.TEST, x_train)

    def __is_required(self, label):
        is_required = True

        if self.labels_requirement == LabelsRequirement.NUMERIC:
            is_required = label.isnumeric()
        elif self.labels_requirement == LabelsRequirement.ALPHA:
            is_required = label.isalpha()

        return is_required

    def __get_accuracy(self, y_test, y_pred, sign):
        correct = 0
        for test, pred in zip(y_test, y_pred):
            if self.__is_prediction_correct(sign, test, pred):
                correct += 1

        return (correct / len(y_test)) * 100

    def __is_prediction_correct(self, sign, test, pred):
        return (sign == test and pred[0] > self.ACCURACY_PERCENTAGE) or \
               (sign != test and pred[0] < self.ACCURACY_PERCENTAGE)
