import os
import numpy as np
from Model.modelEnum import Environment
from Structures.iUtilStructure import Structure
from StrategyFactory.iStrategy import IStrategy
from Structures.NeuralNetworks.neuralNetworkEnum import ClassifierEnum
from Structures.NeuralNetworks.neuralNetworkEnum import LabelsRequirement
from Constraints.path import TMP_BINARY_NEURAL_NETWORK_MODEL_PATH, BINARY_NEURAL_NETWORK_MODEL_PATH


class AccuracyBinaryNeuralNetworkStrategy(IStrategy):
    """
    A class to test the accuracy of a set of binary neural networks models

    Attributes
    ----------
    ACCURACY_PERCENTAGE : number
        Limit set to define that a result is successful
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    nn_util : NeuralNetworkUtil
        A class to execute the common functionalities of a neural network structure
    accuracy_util : AccuracyUtil
        A class to execute the common functionalities in accuracy strategies
    bnn_util : BinaryNeuralNetworkUtil
        A class to execute the common functionalities in the binary neural networks strategies
    storage_controller : StorageController
        A class used to remove and create the directories and files used in the execution
    model_name : string
        Name of the model to test

    Methods
    -------
    execute()
        Show the accuracy of a set of binary neural network models, previously trained, using the test database
    """

    ACCURACY_PERCENTAGE = 0.60

    def __init__(self, logger, model, nn_util, accuracy_util, bnn_util, storage_controller, arguments):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        nn_util : NeuralNetworkUtil
            A class to execute the common functionalities of a neural network structure
        accuracy_util : AccuracyUtil
            A class to execute the common functionalities in accuracy strategies
        bnn_util : BinaryNeuralNetworkUtil
            A class to execute the common functionalities in the binary neural networks strategies
        storage_controller : StorageController
            A class used to remove and create the directories and files used in the execution
        arguments : array
            Array of arguments entered in the execution
        """
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.accuracy_util = accuracy_util
        self.bnn_util = bnn_util
        self.storage_controller = storage_controller

        self.__show_arguments_entered(arguments)

        self.model_name = arguments[0]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Neural Network model file: " + arguments[0]
        self.logger.write_info(info_arguments)

    def execute(self):
        """Show the accuracy of a set of binary neural network models, previously trained, using the test database
        """
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        classifiers, labels_requirement = self.__get_classifiers()

        self.bnn_util.remove_not_wanted_labels(Environment.TEST, labels_requirement)

        self.__perform_test_data(classifiers)

        self.logger.write_info("Strategy executed successfully")

    def __perform_test_data(self, classifiers):
        predictions = []

        self.model.resize_data(Structure.BinaryNeuralNetwork, Environment.TEST)
        x_test = self.model.get_x(Environment.TEST)
        y_test = self.model.get_signs_values(self.model.get_y(Environment.TEST))

        for classifier_dict in classifiers:
            classifier = classifier_dict[ClassifierEnum.CLASSIFIER.value]
            sign = classifier_dict[ClassifierEnum.SIGN.value]

            y_pred = classifier.predict(x_test)
            predictions.append(y_pred)

            self.__show_sign_classifier_accuracy(y_test, y_pred, sign)

        order_signs = np.array([self.model.get_sign_value(sign[ClassifierEnum.SIGN.value]) for sign in classifiers])
        self.__show_global_accuracy(y_test, predictions, order_signs)

    def __get_classifiers(self):

        model_pickles, labels_requirement = self.__get_model_information(self.model_name)
        classifiers = self.__get_classifiers_from_model(self.model_name)
        
        self.__remove_temporal_files(self.model_name)

        self.model.set_pickles_name(model_pickles)

        return classifiers, labels_requirement

    def __get_model_information(self, model_name):
        pickles, labels_requirement = self.nn_util.get_pickles_used_in_binary_zip(model_name)

        return pickles, LabelsRequirement(labels_requirement)

    def __get_classifiers_from_model(self, model_name):
        source_path = BINARY_NEURAL_NETWORK_MODEL_PATH + model_name
        destination_path = TMP_BINARY_NEURAL_NETWORK_MODEL_PATH + model_name[:-len(".zip")]
        model_files_path = self.storage_controller.extract_compressed_files(source_path, destination_path)

        classifiers = []
        for model_path in model_files_path:
            # Get classifier symbol
            sign_array = model_path[:-len(".h5")].split('_')
            sign = sign_array[len(sign_array)-1]

            # Get classifier
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
