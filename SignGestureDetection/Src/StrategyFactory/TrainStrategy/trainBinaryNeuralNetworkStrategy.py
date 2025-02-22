import os
import numpy as np
from Model.model import Model
from Model.modelEnum import Environment
from Storage.storageEnum import FileEnum
from Structures.iUtilStructure import Structure
from StrategyFactory.iStrategy import IStrategy
from Exception.inputOutputException import InputException
from Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil
from Structures.NeuralNetworks.neuralNetworkEnum import LabelsRequirement
from Constraints.path import BINARY_NEURAL_NETWORK_MODEL_PATH, TMP_BINARY_NEURAL_NETWORK_MODEL_PATH
from Structures.NeuralNetworks.NeuralNetworksTypes.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork

class TrainBinaryNeuralNetworkStrategy(IStrategy):
    """
    A class to train a set of binary neural networks models

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    nn_util : NeuralNetworkUtil
        A class to execute the common functionalities of a neural network structure
    bnn_util : BinaryNeuralNetworkUtil
        A class to execute the common functionalities in the binary neural networks strategies
    storage_controller : StorageController
        A class used to remove and create the directories and files used in the execution
    labels_requirement : LabelsRequirement
        Types of labels allowed to reduce the database size
    pickles : array
        Array of pickles' name to use in this strategy

    Methods
    -------
    execute()
        To train a set of binary neural networks models using the training samples of the pickle selected
    """

    def __init__(self, logger, model, nn_util, bnn_util, storage_controller, arguments):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        nn_util : NeuralNetworkUtil
            A class to execute the common functionalities of a neural network structure
        bnn_util : BinaryNeuralNetworkUtil
            A class to execute the common functionalities in the binary neural networks strategies
        storage_controller : StorageController
            A class used to remove and create the directories and files used in the execution
        arguments : array
            Array of arguments entered in the execution
        
        Raises
        ------
        InputException
            If the first argument entered is not a value of the LabelsRequirement enumeration
        """
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.bnn_util = bnn_util
        self.storage_controller = storage_controller
        self.__show_arguments_entered(arguments)

        if arguments[0] not in LabelsRequirement._value2member_map_:
            raise InputException(self.execution_strategy + " is not a valid sign requirement")

        self.labels_requirement = LabelsRequirement(arguments[0])
        self.pickles = arguments[1:]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Signs to train: " + arguments[0] + "\n"\
                         "\t* Pickles selected: " + ", ".join(arguments[1:])
        self.logger.write_info(info_arguments)

    def execute(self):
        """To train a set of binary neural networks models using the training samples of the pickle selected
        """

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.model.set_pickles_name(self.pickles)

        self.bnn_util.remove_not_wanted_labels(Environment.TRAIN, self.labels_requirement)

        self.__prepare_images()
        self.__train_binary_cnn()

        self.logger.write_info("Strategy executed successfully")

    def __prepare_images(self):

        self.model.resize_data(Structure.BinaryNeuralNetwork, Environment.TRAIN)

        x_train = self.model.get_x(Environment.TRAIN).astype('float32')

        self.model.set_x(Environment.TRAIN, x_train)

    def __train_binary_cnn(self):

        classes = np.unique(self.model.get_y(Environment.TRAIN))
        files = []

        for sign in classes:
            self.logger.write_info("Start training the " + sign + " binary classifier")

            cnn = self._init_new_convolution_neural_network_object(sign)
            classifier = cnn.build_sequential_model(1, is_categorical=False)

            file_name = self.__get_sign_model_path(sign)

            file_path = self.__save_model(classifier, file_name)

            files.append({
                FileEnum.FILE_PATH.value: file_path,
                FileEnum.FILE_NAME.value: file_name
            })

        # Save models in a compressed file
        file_path, file_name = self.__get_compressed_file_path()
        self.storage_controller.compress_files(files, file_path + file_name)
        self.storage_controller.remove_files_from_list(files)
        self.nn_util.record_binary_model(file_name, file_path, self.labels_requirement)

    def _init_new_convolution_neural_network_object(self, sign):
        y_train = self.__transform_data(sign)

        model = Model()
        model.set_y(Environment.TRAIN, y_train)
        model.set_x(Environment.TRAIN, self.model.get_x(Environment.TRAIN))
        model.set_pickles_name(self.pickles)

        nn_util = NeuralNetworkUtil(self.logger, model)
        cnn = ConvolutionalNeuralNetwork(self.logger, model, nn_util, improved_nn=True)

        return cnn

    def __transform_data(self, actual_sign):
        y_train = self.model.get_y(Environment.TRAIN)
        indexes = np.where(y_train == actual_sign)[0]
        y_train = np.full(y_train.size, 0)
        y_train[indexes] = 1

        return y_train

    @staticmethod
    def __save_model(classifier, file_name):
        classifier.save(TMP_BINARY_NEURAL_NETWORK_MODEL_PATH + file_name)

        return TMP_BINARY_NEURAL_NETWORK_MODEL_PATH

    @staticmethod
    def __get_sign_model_path(sign):
        return "binary_classifier_" + sign + ".h5"

    def __get_compressed_file_path(self):
        file_name = self.labels_requirement.value + "_" + self.model.get_pickles_name() + "_models.zip"
        return BINARY_NEURAL_NETWORK_MODEL_PATH, file_name
