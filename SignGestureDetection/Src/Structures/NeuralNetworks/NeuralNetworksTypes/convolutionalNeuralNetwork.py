from Model.modelEnum import Environment
from Constraints.hyperparameters import *
from Structures.iUtilStructure import Structure
from tensorflow.keras.constraints import max_norm
from tensorflow.python.keras.models import Sequential
from Exception.parametersException import IncorrectNumberOfParameters
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from Structures.NeuralNetworks.NeuralNetworksTypes.iNeuralNetwork import INeuralNetwork


class ConvolutionalNeuralNetwork(INeuralNetwork):
    """
    A class that contains the convolutional neural network structure and all its functinalities.

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    nn_util : NeuralNetworkUtil
        TODO
    improved_nn : boolean, optional
        Indicates if the model should be build with the improved structure (Defult is false)

    Methods
    -------
    train_neural_network()
        Trains the convolutional neural network model based on the training samples of the database
    prepare_images()
        Process the dataset's samples in order to be ablte to train the model with them
    build_sequential_model(n_classes, is_categorical=True)
        Create sequential model based on f it has to be improved and if it will return a binary or a categorical result
    """

    def __init__(self, logger, model, nn_util, improved_nn=False):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        nn_util : NeuralNetworkUtil
            TODO
        improved_nn : boolean, optional
            Indicates if the model should be build with the improved structure (Defult is false)
        """
        self.logger = logger
        self.model = model
        self.nn_util = nn_util
        self.improved_nn = improved_nn

    def train_neural_network(self):
        """Trains the convolutional neural network model based on the training samples of the database
        """

        n_classes = self.prepare_images()
        sequential_model = self.build_sequential_model(n_classes)

        nn_type = (NeuralNetworkTypeEnum.CNN, NeuralNetworkTypeEnum.IMPROVED_CNN)[self.improved_nn]
        self.nn_util.save_model(sequential_model, nn_type)

    def prepare_images(self):
        """Process the dataset's samples in order to be ablte to train the model with them
        """

        self.model.resize_data(Structure.CategoricalNeuralNetwork, Environment.TRAIN)
        self.model.resize_data(Structure.CategoricalNeuralNetwork, Environment.TEST)

        n_classes = self.model.convert_to_one_hot_data()

        return n_classes

    def build_sequential_model(self, n_classes, is_categorical=True):
        """Load the decision tree trained model and set the dataset used while trianing it

        Parameters
        ----------
        n_classes : number
            Number of different types of classes in the database
        is_categorical : boolean, optional
            Indicates if the model should return a categorical value

        Returns
        -------
        Sequential
            The convolutional neural network model
        """
        shape = self.model.get_x(Environment.TRAIN).shape

        if self.improved_nn:
            seq_model = self.__get_improved_sequential_model(n_classes, shape, is_categorical)
            seq_model = self.nn_util.train_model(seq_model, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                                 is_categorical=is_categorical)

        else:
            seq_model = self.__get_not_improved_sequential_model(n_classes, shape)
            seq_model = self.nn_util.train_model(seq_model)

        return seq_model

    @staticmethod
    def __get_improved_sequential_model(n_classes, shape, is_categorical):
        output_layer_activation = ('sigmoid', 'softmax')[is_categorical]

        sequential_model = Sequential()
        sequential_model.add(Conv2D(NEURONS_CONV_LAYER, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                    activation=ACTIVATION, input_shape=(shape[1], shape[2], 1),
                                    kernel_initializer=INIT_MODE, kernel_constraint=max_norm(WEIGHT_CONSTRAINT)))
        sequential_model.add(MaxPool2D(pool_size=(1, 1)))
        sequential_model.add(Flatten())
        sequential_model.add(Dense(NEURONS_DENSE_LAYER, kernel_initializer=INIT_MODE, activation=ACTIVATION))
        sequential_model.add(Dense(n_classes, activation=output_layer_activation))

        return sequential_model

    @staticmethod
    def __get_not_improved_sequential_model(n_classes, shape):
        # building a linear stack of layers with the sequential model
        sequential_model = Sequential()

        # Input layer: convolutional layer
        sequential_model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                                    input_shape=(shape[1], shape[2], 1)))

        # Hidden layer: pooling Layers: Prevent overfitting
        sequential_model.add(MaxPool2D(pool_size=(1, 1)))

        # Hidden layer: flatten output of conv
        sequential_model.add(Flatten())

        # Hidden layer
        sequential_model.add(Dense(100, activation='relu'))

        # Output layer
        sequential_model.add(Dense(n_classes, activation='softmax'))

        return sequential_model
