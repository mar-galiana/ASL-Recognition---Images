import os
from tensorflow import keras
from Model.modelEnum import Environment
from Exception.modelException import EnvironmentException
from Structures.iUtilStructure import IUtilStructure, Structure
from Constraints.path import TMP_BINARY_NEURAL_NETWORK_MODEL_PATH
from Constraints.path import CATEGORICAL_NEURAL_NETWORK_MODEL_PATH
from Exception.inputOutputException import PathDoesNotExistException
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum


class NeuralNetworkUtil(IUtilStructure):
    """
    A class to execute the common functionalities of a neural network structure.

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database

    Methods
    -------
    train_model(sequential_model, batch_size=128, epochs=10, is_categorical=False)
        Train the neural network model based on the training samples of the database
    save_model(model, neural_network_type)
        Store the categorical neural network trained model into a file with extension ".h5"
    load_model(name_model)
        Load the neural network trained model and set the dataset used while trianing it 
    get_pickles_used_in_binary_zip(name_model)
        Get the pickles used to train a binary neural network model
    record_binary_model(file_name, file_path, restriction)
        Store the categorical neural network information while training it in the json file
    read_model(nn_model_path)
        Load the keras model
    """

    def __init__(self, logger, model):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        """

        self.logger = logger
        self.model = model

    def train_model(self, sequential_model, batch_size=128, epochs=10, is_categorical=False):
        """Train the neural network model based on the training samples of the database

        Parameters
        ----------
        sequential_model : Sequential
            The neural network model without training it
        batch_size : number, optional
            The batch size (Default is 128)
        epochs : number, optional
            The number of epochs (Default is 10)
        is_categorical : boolean, optional
            Indicates if the model should return a categorical value (Default is False)

        Returns
        -------
        Sequential
            The neural network model trained
        """

        loss = ('binary_crossentropy', 'categorical_crossentropy')[is_categorical]

        sequential_model.summary()

        # compiling the sequential model
        sequential_model.compile(loss=loss, metrics=['accuracy'], optimizer='adam')

        # training the model
        sequential_model.fit(self.model.get_x(Environment.TRAIN), self.model.get_y(Environment.TRAIN),
                             batch_size=batch_size, epochs=epochs)

        return sequential_model

    def save_model(self, model, neural_network_type):
        """Store the categorical neural network trained model into a file with extension ".h5"

        Parameters
        ----------
        model : Sequential
            Model to save
        neural_network_type : NeuralNetworkTypeEnum
            Types of neural networks
        """
        model_path, model_name = self.__get_keras_model_path(neural_network_type)

        model.save(model_path + model_name)

        super(NeuralNetworkUtil, self).save_pickles_used(Structure.CategoricalNeuralNetwork,
                                                         self.model.get_pickles_name(),
                                                         model_name)

        self.logger.write_info("A new categorical neural network model has been created with the name of: " + model_name
                               + "\nIn the path: " + model_path + "\nThis is the name that will be needed in the "
                               "other strategies if you want to work with this model.")

    def load_model(self, name_model):
        """Load the neural network trained model and set the dataset used while trianing it

        Parameters
        ----------
        name_model : string
            Name of the model to load

        Raises
        ------
        PathDoesNotExistException
            If the model's name does not exist

        Returns
        -------
        Sequential
            The neural network model
        NeuralNetworkTypeEnum
            Types of neural networks
        """

        nn_model_path = CATEGORICAL_NEURAL_NETWORK_MODEL_PATH + name_model

        if not os.path.exists(nn_model_path):
            raise PathDoesNotExistException("The model needs to exists to be able to use it")

        pickles, nn_type = super(NeuralNetworkUtil, self).get_pickles_used(Structure.CategoricalNeuralNetwork,
                                                                           name_model)
        self.model.set_pickles_name(pickles)

        keras_model = self.read_model(nn_model_path)

        return keras_model, NeuralNetworkTypeEnum(nn_type)

    def get_pickles_used_in_binary_zip(self, name_model):
        """Get the pickles used to train a binary neural network model

        Parameters
        ----------
        name_model : string
            Name of the model to load

        Returns
        -------
        array
            Pickles used in the training
        """

        pickles = super(NeuralNetworkUtil, self).get_pickles_used(Structure.BinaryNeuralNetwork, name_model)
        return pickles

    def record_binary_model(self, file_name, file_path, restriction):
        """Store the categorical neural network information while training it in the json file

        Parameters
        ----------
        file_name : string
            Name of the file created
        file_path : string
            Path of the file created
        restriction : LabelsRequirement
            Types of labels allowed to reduce the database size
        """

        super(NeuralNetworkUtil, self).save_pickles_used(Structure.BinaryNeuralNetwork, self.model.get_pickles_name(),
                                                         file_name, restriction=restriction)

        self.logger.write_info("A new set of binary neural network models have been created with the name of: " +
                               file_name + "\nIn the path: " + file_path + "\nThis is the name that will be needed in "
                               "the other strategies if you want to work with these models.")

    @staticmethod
    def read_model(nn_model_path):
        """Load the keras model

        Parameters
        ----------
        nn_model_path : string
            Path of the model to load

        Returns
        -------
        Sequential
            Neural network model
        """
        return keras.models.load_model(nn_model_path)

    def __get_keras_model_path(self, neural_network_type):
        if not isinstance(neural_network_type, NeuralNetworkTypeEnum):
            raise EnvironmentException("Environment used is not a valid one")

        file_name = neural_network_type.value + "_" + self.model.get_pickles_name() + "_model"

        return CATEGORICAL_NEURAL_NETWORK_MODEL_PATH, file_name + ".h5"
