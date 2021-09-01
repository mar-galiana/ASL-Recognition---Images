import os
import numpy as np
from skimage import io
from skimage.transform import resize
from Model.enumerations import Environment
from StrategyFactory.iStrategy import IStrategy
from tensorflow.python.keras.preprocessing import image
from Exception.inputOutputException import InputException
from Structures.NeuralNetworks.neuralNetwork import NeuralNetwork
from Structures.NeuralNetworks.enumerations import NeuralNetworkEnum
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class AccuracyBinaryClassificationStrategy(IStrategy):

    def __init__(self, logger, model, nn_util, accuracy_util, arguments):
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.accuracy_util = accuracy_util

        self.__show_arguments_entered(arguments)

        self.name_nn_model = arguments[0]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Neural Network model file: " + arguments[0]
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.predict()
        self.logger.write_info("Strategy executed successfully")

    def predict(self):

        with zipfile.ZipFile(BINARY_NEURAL_NETWORKS_MODELS_PATH + "asl_alphabet_gray_150x150px_models.zip", "r") as zip_ref:
            zip_ref.extractall(TMP_BINARY_NEURAL_NETWORKS_MODELS_PATH)

        for f in os.listdir(TMP_BINARY_NEURAL_NETWORKS_MODELS_PATH + "asl_alphabet_gray_150x150px_models"):
            if not f.endswith(".h5"):
                continue
            print(f)

        for f in os.listdir(TMP_BINARY_NEURAL_NETWORKS_MODELS_PATH + "asl_alphabet_gray_150x150px_models"):
            if not f.endswith(".h5"):
                continue
            os.remove(os.path.join(TMP_BINARY_NEURAL_NETWORKS_MODELS_PATH, "asl_alphabet_gray_150x150px_models", f))
        os.rmdir(os.path.join(TMP_BINARY_NEURAL_NETWORKS_MODELS_PATH, "asl_alphabet_gray_150x150px_models"))