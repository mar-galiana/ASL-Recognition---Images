import os
import numpy as np
from skimage import io
from skimage.transform import resize
from Model.modelEnum import Environment
from Structures.iUtilStructure import Structure
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

    ACCURACY_PERCENTAGE = 0.60

    def __init__(self, logger, model, nn_util, accuracy_util, bnn_util, storage_controller, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.accuracy_util = accuracy_util
        self.bnn_util = bnn_util
        self.storage_controller = storage_controller

        self.__show_arguments_entered(arguments)

        if arguments[0] not in LabelsRequirement._value2member_map_:
            raise InputException(self.execution_strategy + " is not a valid sign requirement")

        self.labels_requirement = LabelsRequirement(arguments[0])
        self.models_names = arguments[1:]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Signs to train: " + arguments[0] + "\n" \
                         "\t* Neural Network model file: " + ", ".join(arguments[1:])
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        classifiers = self.__get_classifiers()

        self.bnn_util.remove_not_wanted_labels(Environment.TEST, self.labels_requirement)

        self.perform_test_data(classifiers)

        self.logger.write_info("Strategy executed successfully")

    def perform_test_data(self, classifiers):
        predictions = []

        shape = self.model.get_x(Environment.TEST).shape

        self.model.resize_data(Structure.BinaryNeuralNetwork, Environment.TEST, shape)
        x_test = self.model.get_x(Environment.TEST)
        y_test = self.model.get_sign_values(self.model.get_y(Environment.TEST))

        for classifier_dict in classifiers:
            classifier = classifier_dict[ClassifierEnum.CLASSIFIER.value]
            sign = classifier_dict[ClassifierEnum.SIGN.value]

            y_pred = classifier.predict(x_test)
            predictions.append(y_pred)

            self.__show_sign_classifier_accuracy(y_test, y_pred, sign)

        order_signs = np.array([self.model.get_sign_value(sign[ClassifierEnum.SIGN.value]) for sign in classifiers])
        self.__show_global_accuracy(y_test, predictions, order_signs)

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

    def __remove_temporal_files(self, model_name):
        directory_path = TMP_BINARY_NEURAL_NETWORK_MODEL_PATH + model_name[:-len(".zip")]
        self.storage_controller.remove_files_from_folder(directory_path)

    def __show_sign_classifier_accuracy(self, y_test, y_pred, sign):
        numeric_sign = self.model.get_sign_value(sign)
        correct = 0
        a = 0

        for test, pred in zip(y_test, y_pred):
            a += 1
            if self.__is_prediction_correct(numeric_sign, test, pred):
                correct += 1

        accuracy = (correct / len(y_test)) * 100

        self.logger.write_info("The accuracy of the binary neural network of the sign '" + sign + "' is: "
                               "{:.2f}".format(accuracy) + "%")

    def __show_global_accuracy(self, y_test, predictions, signs_prediction):
        predictions = np.array(predictions)
        correct, more_than_one_solution = 0, 0

        for index, test in enumerate(y_test):

            indices = self.__get_max_value_indices(predictions[:, index])
            is_correct, is_more_than_one_correct = self.__is_global_prediction_correct(test, indices, signs_prediction)

            if is_correct:
                correct += 1

            if is_more_than_one_correct:
                more_than_one_solution += 1

        correct_accuracy = (correct / len(y_test)) * 100
        more_than_one_solution_accuracy = (more_than_one_solution / len(y_test)) * 100

        self.logger.write_info("The global accuracy is: {:.2f}".format(correct_accuracy) + "%")
        self.logger.write_info("{:.2f}".format(more_than_one_solution_accuracy) + "% of the correct samples had more "
                               "than one solution.")

    @staticmethod
    def __get_max_value_indices(predictions):
        max_value = -1
        indices = []

        for index, prediction in enumerate(predictions):
            if prediction > max_value:
                max_value = prediction
                indices = [index]
            elif prediction == max_value:
                indices.append(index)

        return indices

    @staticmethod
    def __is_global_prediction_correct(test_label, indices_prediction,  order_signs):
        correct = False

        for index in indices_prediction:
            if test_label != order_signs[index]:
                continue

            correct = True
            break

        more_than_one_solution = correct and len(indices_prediction) > 1

        return correct, more_than_one_solution

    def __is_prediction_correct(self, sign, test, pred):
        return (sign == test and pred[0] > self.ACCURACY_PERCENTAGE) or \
               (sign != test and pred[0] < 1 - self.ACCURACY_PERCENTAGE)
