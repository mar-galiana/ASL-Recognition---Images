import os
from StrategyFactory.iStrategy import IStrategy
from Exception.inputOutputException import InputException
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Structures.NeuralNetworks.NeuralNetworksTypes.artificialNeuralNetwork import ArtificialNeuralNetwork
from Structures.NeuralNetworks.NeuralNetworksTypes.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class TrainCategoricalNeuralNetworkStrategy(IStrategy):
    """
    A class to train a categorical neural network model

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    nn_util : NeuralNetworkUtil
        A class to execute the common functionalities of a neural network structure
    nn_type : string
        Type of neural network, it has to be a value of the NeuralNetworkTypeEnum enumerator
    pickles : array
        Array of pickles to use in the training
    algorithm_switcher : dictionary
        Dictionary that contains the model class that will execute the training depending on the neural network type

    Methods
    -------
    execute()
        To train a categorical neural network model using the training samples of the pickle selected
    """

    def __init__(self, logger, model, nn_util, arguments):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        nn_util : NeuralNetworkUtil
            A class to execute the common functionalities of a neural network structure
        arguments : array
            Array of arguments entered in the execution
        """
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.__show_arguments_entered(arguments)

        self.nn_type = arguments[0]
        self.pickles = arguments[1:]

        self.algorithm_switcher = {
            NeuralNetworkTypeEnum.ANN.value: ArtificialNeuralNetwork(self.logger, self.model, self.nn_util),
            NeuralNetworkTypeEnum.CNN.value: ConvolutionalNeuralNetwork(self.logger, self.model, self.nn_util),
            NeuralNetworkTypeEnum.IMPROVED_CNN.value: ConvolutionalNeuralNetwork(self.logger, self.model, self.nn_util,
                                                                                 improved_nn=True)
        }

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Neural Network type: " + arguments[0] + "\n" \
                         "\t* Pickles selected: " + ", ".join(arguments[1:])
        self.logger.write_info(info_arguments)

    def execute(self):
        """To train a categorical neural network model using the training samples of the pickle selected

        Raises
        ------
        InputException
            If the neural network type is not a value of the NeuralNetworkTypeEnum enumeration
        """

        if self.nn_type not in self.algorithm_switcher:
            raise InputException(self.nn_type + " is not a valid strategy")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.model.set_pickles_name(self.pickles)

        algorithm_execution = self.algorithm_switcher.get(self.nn_type)
        algorithm_execution.train_neural_network()

        self.logger.write_info("Strategy executed successfully")
