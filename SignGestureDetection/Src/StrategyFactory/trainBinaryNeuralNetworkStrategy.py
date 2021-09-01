import os
import numpy as np
from Src.Model.model import Model
from Src.Storage.storageEnum import FileEnum
from Model.enumerations import Environment
from Src.StrategyFactory.iStrategy import IStrategy
from tensorflow.python.keras.models import Sequential
from sklearn.preprocessing import MultiLabelBinarizer
from Src.Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil
from Src.Constraints.path import BINARY_CNN_MODEL_PATH, TMP_BINARY_CNN_MODEL_PATH
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from Src.Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class TrainBinaryNeuralNetworkStrategy(IStrategy):

    def __init__(self, logger, model, nn_util, storage_controller, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.storage_controller = storage_controller
        self.__show_arguments_entered(arguments)

        self.pickels = arguments

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Pickels selected: " + ", ".join(arguments)
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.model.set_pickels_name(self.pickels)

        self.__remove_not_wanted_labels(Environment.TRAIN)
        self.__remove_not_wanted_labels(Environment.TEST)
        self.__prepare_images()
        self.__train_binary_cnn()

        self.logger.write_info("Strategy executed successfully")

    def __resize_data(self, environment, shape):
        x_data = self.model.get_x(environment).reshape(shape[0], shape[1], shape[2], 1)
        return x_data

    def __prepare_images(self):
        shape_train = self.model.get_x(Environment.TRAIN).shape
        shape_test = self.model.get_x(Environment.TEST).shape

        x_train = self.__resize_data(Environment.TRAIN, shape_train).astype('float32')
        x_test = self.__resize_data(Environment.TEST, shape_test).astype('float32')

        self.model.set_x(Environment.TRAIN, x_train)
        self.model.set_x(Environment.TEST, x_test)

    def __train_binary_cnn(self):

        classes = np.unique(self.model.get_y(Environment.TRAIN))
        files = []

        for sign in classes:
            self.logger.write_info("Start training the " + sign + " binary classifier")

            cnn, nn_util = self._init_new_convolution_neural_network_object(sign)
            classifier = cnn.build_sequential_model(1, self.model.get_x(Environment.TRAIN).shape, is_categorical=True)

            file_name = self.__get_sign_model_path(sign)

            file_path = self.__save_model(classifier, file_name)

            files.append({
                FileEnum.FILE_PATH.value: file_path,
                FileEnum.FILE_NAME.value: file_name
            })

        file_path, file_name = self.__get_compressed_file_path()
        self.storage_controller.compress_files(files, file_path + file_name)
        self.storage_controller.remove_files_from_folder(files)
        self.nn_util.record_binary_model(file_name, file_path)

    def _init_new_convolution_neural_network_object(self, sign):
        y_train = self.__transform_data(sign)

        model = Model()
        model.set_y(Environment.TRAIN, y_train)
        model.set_x(Environment.TRAIN, self.model.get_x(Environment.TRAIN))
        model.set_pickels_name(self.pickels)

        nn_util = NeuralNetworkUtil(self.logger, model)
        cnn = ConvolutionalNeuralNetwork(self.logger, model, nn_util, True)

        return cnn, nn_util

    def __transform_data(self, actual_sign):
        y_train = self.model.get_y(Environment.TRAIN)
        indexes = np.where(y_train == actual_sign)[0]
        y_train = np.full(y_train.size, 0)
        y_train[indexes] = 1

        return y_train

    def __remove_not_wanted_labels(self, environment):
        y_train = self.model.get_y(environment)
        x_train = self.model.get_x(environment)
        indexes = [i for i, label in enumerate(y_train) if label != 'A' and label != 'B' and label != 'C']
        y_train = np.delete(y_train, indexes)
        x_train = np.delete(x_train, indexes, axis=0)

        self.model.set_y(environment, y_train)
        self.model.set_x(environment, x_train)

    @staticmethod
    def __save_model(classifier, file_name):
        classifier.save(TMP_BINARY_CNN_MODEL_PATH + file_name)

        return TMP_BINARY_CNN_MODEL_PATH

    @staticmethod
    def __get_sign_model_path(sign):
        return "binary_classifier_" + sign + ".h5"

    def __get_compressed_file_path(self):
        file_name = self.model.get_pickels_name() + "_models.zip"
        return BINARY_CNN_MODEL_PATH, file_name
