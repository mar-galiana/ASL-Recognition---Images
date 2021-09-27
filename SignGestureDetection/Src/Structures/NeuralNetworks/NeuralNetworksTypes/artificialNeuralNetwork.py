from Model.modelEnum import Environment
from Structures.iUtilStructure import Structure
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Structures.NeuralNetworks.NeuralNetworksTypes.iNeuralNetwork import INeuralNetwork


class ArtificialNeuralNetwork(INeuralNetwork):
    """
    A class that contains the artificial neural network structure and all its functinalities.

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    nn_util : NeuralNetworkUtil
        TODO

    Methods
    -------
    train_neural_network()
        Trains the artificial neural network model based on the training samples of the database
    """

    def __init__(self, logger, model, nn_util):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        nn_util : NeuralNetworkUtil
            TODO
        """
        self.model = model
        self.logger = logger
        self.nn_util = nn_util

    def train_neural_network(self):
        """Trains the artificial neural network model based on the training samples of the database
        """
        n_classes = self.__prepare_images()
        sequential_model = self.__build_sequential_model(n_classes)
        self.nn_util.save_model(sequential_model, NeuralNetworkTypeEnum.ANN)

    def __prepare_images(self):
        self.model.resize_data(Structure.CategoricalNeuralNetwork, Environment.TRAIN, NeuralNetworkTypeEnum.ANN)
        self.model.resize_data(Structure.CategoricalNeuralNetwork, Environment.TEST, NeuralNetworkTypeEnum.ANN)

        n_classes = self.model.convert_to_one_hot_data()
        return n_classes

    def __build_sequential_model(self, n_classes):
        shape_train = self.model.get_x(Environment.TRAIN).shape

        sequential_model = Sequential()
       
        # Input layer
        sequential_model.add(Dense(100, input_shape=(shape_train[1]*shape_train[2],), activation='relu'))
       
        # Hidden layer
        sequential_model.add(Dense(n_classes, activation='softmax'))
        
        # Output layer
        sequential_model = self.nn_util.train_model(sequential_model)
        return sequential_model
