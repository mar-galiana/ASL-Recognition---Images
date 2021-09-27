from Model.modelEnum import Environment
from Constraints.hyperparameters import *
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.constraints import max_norm
from tensorflow.python.keras.models import Sequential
from Exception.parametersException import IncorrectVariableType
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from Structures.NeuralNetworks.neuralNetworkEnum import AttributeToTuneEnum
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from Structures.NeuralNetworks.NeuralNetworksTypes.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class HyperparameterOptimization:
    """
    A class to get the optimal value for the hyperparameters of a convolutional neural network model.

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    nn_util : NeuralNetworkUtil
        A class to execute the common functionalities of a neural network structure
    cnn : ConvolutionalNeuralNetwork
        A class that contains the convolutional neural network structure and all its functionalities
    parameters_switcher : dictionary
        Dictionary that will return the function implementing the optimizer module based on the AttributeToTuneEnum
        enumerator
    
    Methods
    -------
    calculate_best_hyperparameter_optimization(attribute_tune)
        Calculate the best value for the CNN module based on the attribute selected
    """
    
    def __init__(self, logger, model, nn_util):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        nn_util : NeuralNetworkUtil
            A class to execute the common functionalities of a neural network structure
        """

        self.model = model
        self.logger = logger
        self.nn_util = nn_util
        self.cnn = ConvolutionalNeuralNetwork(logger, model, nn_util)

        self.parameters_switcher = self.__get_parameter_switcher()

    def calculate_best_hyperparameter_optimization(self, attribute_tune):
        """Calculate the best value for the CNN module based on the attribute selected

        Parameters
        ----------
        attribute_tune : AttributeToTuneEnum
            Attribute to be optimized

        Raises
        ------
        IncorrectVariableType
            If the attribute_tune variable is not an AttributeToTuneEnum enumeration
        """

        if not isinstance(attribute_tune, AttributeToTuneEnum):
            raise IncorrectVariableType("Expecting AttributeToTuneEnum enumeration")

        n_classes, image_size = self.__prepare_data()
        grid = self.__get_grid_search_classifier(attribute_tune, n_classes, image_size)
        grid_result = self.__train_convolutional_neural_network(grid)
        
        self.__summarize_results(grid_result)

    def __get_grid_search_classifier(self, attribute_tune, n_classes, image_size):
        parameters_function = self.parameters_switcher.get(attribute_tune)
        param_grid, classifier = parameters_function(n_classes, image_size)
        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=1, cv=3)
        return grid

    def __prepare_data(self):
        shape_train = self.model.get_x(Environment.TRAIN).shape
        shape_test = self.model.get_x(Environment.TEST).shape
        n_classes = self.cnn.prepare_images(shape_train, shape_test)
        return n_classes, shape_train[1:]

    def __train_convolutional_neural_network(self, grid):
        x_train = self.model.get_x(Environment.TRAIN)
        y_train = self.model.get_y(Environment.TRAIN)
        grid_result = grid.fit(x_train, y_train)
        return grid_result

    @staticmethod
    def __get_parameters_for_batch_epochs(n_classes, image_size):
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100]

        param_grid = dict(batch_size=batch_size, epochs=epochs, num_classes=[n_classes], image_size=[image_size])
        classifier = KerasClassifier(build_fn=create_model_batch_epochs, verbose=2)

        return param_grid, classifier

    @staticmethod
    def __get_parameters_for_optimization_algorithm(n_classes, image_size):
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

        param_grid = dict(optimizer=optimizer, num_classes=[n_classes], image_size=[image_size])
        classifier = KerasClassifier(build_fn=create_model_optimizer_algorithm, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                     verbose=2)

        return param_grid, classifier

    @staticmethod
    def __get_parameters_for_learn_rate_and_momentum(n_classes, image_size):
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

        param_grid = dict(learn_rate=learn_rate, momentum=momentum, num_classes=[n_classes], image_size=[image_size])
        classifier = KerasClassifier(build_fn=create_model_learn_rate_and_momentum, epochs=EPOCHS,
                                     batch_size=BATCH_SIZE, verbose=2)

        return param_grid, classifier

    @staticmethod
    def __get_parameters_network_weight_init(n_classes, image_size):
        init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                     'he_uniform']

        param_grid = dict(init_mode=init_mode, num_classes=[n_classes], image_size=[image_size])
        classifier = KerasClassifier(build_fn=create_model_network_weight_init, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                     verbose=2)

        return param_grid, classifier

    @staticmethod
    def __get_parameters_neuron_activation_function(n_classes, image_size):
        activation = ['relu', 'softmax', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

        param_grid = dict(activation=activation, num_classes=[n_classes], image_size=[image_size])
        classifier = KerasClassifier(build_fn=create_model_neuron_activation_function, epochs=EPOCHS,
                                     batch_size=BATCH_SIZE, verbose=2)

        return param_grid, classifier

    @staticmethod
    def __get_parameters_dropout_regularization(n_classes, image_size):
        weight_constraint = [1, 2, 3, 4, 5]
        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint, num_classes=[n_classes],
                          image_size=[image_size])
        classifier = KerasClassifier(build_fn=create_model_dropout_regularization, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                     verbose=2)

        return param_grid, classifier

    @staticmethod
    def __get_parameters_number_neurons(n_classes, image_size):
        neurons_conv_layer = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 140]
        neurons_dense_layer = [10, 20, 25, 30, 40, 50, 60, 80, 100, 120, 140]

        param_grid = dict(neurons_conv_layer=neurons_conv_layer, neurons_dense_layer=neurons_dense_layer,
                          num_classes=[n_classes], image_size=[image_size])
        classifier = KerasClassifier(build_fn=create_model_number_neurons, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                     verbose=2)

        return param_grid, classifier

    def __summarize_results(self, grid_result):
        self.logger.write_info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for mean, stdev, param in zip(means, stds, params):
            self.logger.write_info("%f (%f) with: %r" % (mean, stdev, param))

    def __get_parameter_switcher(self):
        return {
            AttributeToTuneEnum.BATCH_SIZE_AND_EPOCHS:
                lambda n, size: self.__get_parameters_for_batch_epochs(n, size),
            AttributeToTuneEnum.OPTIMIZATION_ALGORITHMS:
                lambda n, size: self.__get_parameters_for_optimization_algorithm(n, size),
            AttributeToTuneEnum.LEARN_RATE_AND_MOMENTUM:
                lambda n, size: self.__get_parameters_for_learn_rate_and_momentum(n, size),
            AttributeToTuneEnum.NETWORK_WEIGHT_INITIALIZATION:
                lambda n, size: self.__get_parameters_network_weight_init(n, size),
            AttributeToTuneEnum.NEURON_ACTIVATION_FUNCTION:
                lambda n, size: self.__get_parameters_neuron_activation_function(n, size),
            AttributeToTuneEnum.DROPOUT_REGULARIZATION:
                lambda n, size: self.__get_parameters_dropout_regularization(n, size),
            AttributeToTuneEnum.NUMBER_NEURONS:
                lambda n, size: self.__get_parameters_number_neurons(n, size)
        }


def get_default_sequential_model(num_classes, image_size):
    """Calculate the best value for the CNN module based on the attribute selected

    Parameters
    ----------
    num_classes : number
        Number of different types of classes in the database selected to train the model
    image_size : tuple
        Image shape

    Returns
    -------
    Sequential
        The convolutional neural network model to compile and train
    """
    model = Sequential()
    model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                     input_shape=(image_size[0], image_size[1], 1)))
    model.add(MaxPool2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def create_model_batch_epochs(num_classes=39, image_size=(150, 150)):
    """Calculate the best values for the batch and epoch hyperparameters

    Parameters
    ----------
    num_classes : number, optional
        Number of different types of classes in the database selected to train the model (Default is 39)
    image_size : tuple, optional
        Image shape (Default is (150, 150))

    Returns
    -------
    Sequential
        The convolutional neural network model to train
    """

    # create model
    model = get_default_sequential_model(num_classes, image_size)

    # Compile model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_model_optimizer_algorithm(optimizer='adam', num_classes=39, image_size=(150, 150)):
    """Calculate the best value for the optimization algorithm hyperparameter

    Parameters
    ----------
    optimizer : string, optional
        Optimizer algorithm value (Default is "adam")
    num_classes : number, optional
        Number of different types of classes in the database selected to train the model (Default is 39)
    image_size : tuple, optional
        Image shape (Default is (150, 150))

    Returns
    -------
    Sequential
        The convolutional neural network model to train
    """

    # create model
    model = get_default_sequential_model(num_classes, image_size)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_learn_rate_and_momentum(learn_rate=0.01, momentum=0, num_classes=39, image_size=(150, 150)):
    """Calculate the best values for the learn rate and momentum hyperparameters

    Parameters
    ----------
    learn_rate : string, optional
        Learn rate value (Default is 0.01)
    momentum : number, optional
        Momentum value (Default is 0)
    num_classes : number, optional
        Number of different types of classes in the database selected to train the model (Default is 39)
    image_size : tuple, optional
        Image shape (Default is (150, 150))

    Returns
    -------
    Sequential
        The convolutional neural network model to train
    """

    # create model
    model = get_default_sequential_model(num_classes, image_size)

    # Compile model
    optimizer = SGD(learning_rate=learn_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_network_weight_init(init_mode='uniform', num_classes=39, image_size=(150, 150)):
    """Calculate the best value for the network weight initialization hyperparameter

    Parameters
    ----------
    init_mode : string, optional
        Network weight initialization value (Default is "uniform")
    num_classes : number, optional
        Number of different types of classes in the database selected to train the model (Default is 39)
    image_size : tuple, optional
        Image shape (Default is (150, 150))

    Returns
    -------
    Sequential
        The convolutional neural network model to train
    """

    # create model
    model = Sequential()
    model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                     input_shape=(image_size[0], image_size[1], 1), kernel_initializer=init_mode))
    model.add(MaxPool2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    optimizer = SGD(learning_rate=LEARN_RATE, momentum=MOMENTUM)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_neuron_activation_function(activation='relu', num_classes=39, image_size=(150, 150)):
    """Calculate the best value for the neuron activation function hyperparameter

    Parameters
    ----------
    activation : string, optional
        Neuron activation function value (Default is "relu")
    num_classes : number, optional
        Number of different types of classes in the database selected to train the model (Default is 39)
    image_size : tuple, optional
        Image shape (Default is (150, 150))

    Returns
    -------
    Sequential
        The convolutional neural network model to train
    """

    # create model
    model = Sequential()
    model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=activation,
                     input_shape=(image_size[0], image_size[1], 1), kernel_initializer=INIT_MODE))
    model.add(MaxPool2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer='uniform', activation=activation))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    optimizer = SGD(learning_rate=LEARN_RATE, momentum=MOMENTUM)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_dropout_regularization(dropout_rate=0.0, weight_constraint=0, num_classes=39, image_size=(150, 150)):
    """Calculate the best values for the dropout rate and weight constraint hyperparameters

    Parameters
    ----------
    dropout_rate : number, optional
        Dropout rate value (Default is 0.0)
    weight_constraint : number, optional
        Weight constraint value (Default is 0)
    num_classes : number, optional
        Number of different types of classes in the database selected to train the model (Default is 39)
    image_size : tuple, optional
        Image shape (Default is (150, 150))

    Returns
    -------
    Sequential
        The convolutional neural network model to train
    """

    # create model
    model = Sequential()
    model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=ACTIVATION,
                     input_shape=(image_size[0], image_size[1], 1), kernel_initializer=INIT_MODE,
                     kernel_constraint=max_norm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer='uniform', activation=ACTIVATION))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    optimizer = SGD(learning_rate=LEARN_RATE, momentum=MOMENTUM)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def create_model_number_neurons(neurons_conv_layer=25, neurons_dense_layer=100, num_classes=39, image_size=(150, 150)):
    """Calculate the best values for the number of neurons of the convolutional and the dense layer

    Parameters
    ----------
    neurons_conv_layer : number, optional
        Neurons of the convolutional layer (Default is 25)
    neurons_dense_layer : number, optional
        Neurons of the dense layer (Default is 100)
    num_classes : number, optional
        Number of different types of classes in the database selected to train the model (Default is 39)
    image_size : tuple, optional
        Image shape (Default is (150, 150))

    Returns
    -------
    Sequential
        The convolutional neural network model to train
    """

    # create model
    model = Sequential()
    model.add(Conv2D(neurons_conv_layer, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=ACTIVATION,
                     input_shape=(image_size[0], image_size[1], 1), kernel_initializer=INIT_MODE,
                     kernel_constraint=max_norm(WEIGHT_CONSTRAINT)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(MaxPool2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(neurons_dense_layer, kernel_initializer='uniform', activation=ACTIVATION))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    optimizer = SGD(learning_rate=LEARN_RATE, momentum=MOMENTUM)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
