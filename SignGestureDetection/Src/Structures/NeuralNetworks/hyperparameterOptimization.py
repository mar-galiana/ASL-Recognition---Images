"""
By default, the grid search will only use one thread. By setting the n_jobs argument in the GridSearchCV constructor to
-1, the process will use all cores on your machine

The best_score_ member provides access to the best score observed during the optimization procedure and the best_params_
describes the combination of parameters that achieved the best results.
"""
from Assets.hyperparameters import *
from Model.enumerations import Environment
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.constraints import max_norm
from tensorflow.python.keras.models import Sequential
from Exception.structureException import IncorrectVariableType
from Structures.NeuralNetworks.enumerations import AttributeToTune
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


def get_default_sequential_model(num_classes, image_size):
    model = Sequential()
    model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                     input_shape=(image_size[0], image_size[1], 1)))
    model.add(MaxPool2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def create_model_batch_epochs(num_classes=39, image_size=(150, 150)):
    # create model
    model = get_default_sequential_model(num_classes, image_size)

    # Compile model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_model_optimizer_algorithm(optimizer='adam', num_classes=39, image_size=(150, 150)):
    # create model
    model = get_default_sequential_model(num_classes, image_size)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_learn_rate_and_momentum(learn_rate=0.01, momentum=0, num_classes=39, image_size=(150, 150)):
    # create model
    model = get_default_sequential_model(num_classes, image_size)

    # Compile model
    optimizer = SGD(learning_rate=learn_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_network_weight_init(init_mode='uniform', num_classes=39, image_size=(150, 150)):
    # create model
    model = Sequential()
    model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                     input_shape=(image_size[0], image_size[1], 1), kernel_initializer=init_mode))
    model.add(MaxPool2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    optimizer = SGD(learning_rate=LEARN_RATE, momentum=MOMENTUM)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_neuron_activation_function(activation='relu', num_classes=39, image_size=(150, 150)):
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


class HyperparameterOptimization:

    def __init__(self, logger, model, nn_util):
        self.model = model
        self.logger = logger
        self.nn_util = nn_util
        self.cnn = ConvolutionalNeuralNetwork(logger, model, nn_util)

        self.parameters_switcher = self.__get_parameter_switcher()

    def calculate_best_hyperparameter_optimization(self, attribute_tune):

        if not isinstance(attribute_tune, AttributeToTune):
            raise IncorrectVariableType("Expecting AttributeToTune enumeration")

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
            AttributeToTune.BATCH_SIZE_AND_EPOCHS:
                lambda n, size: self.__get_parameters_for_batch_epochs(n, size),
            AttributeToTune.OPTIMIZATION_ALGORITHMS:
                lambda n, size: self.__get_parameters_for_optimization_algorithm(n, size),
            AttributeToTune.LEARN_RATE_AND_MOMENTUM:
                lambda n, size: self.__get_parameters_for_learn_rate_and_momentum(n, size),
            AttributeToTune.NETWORK_WEIGHT_INITIALIZATION:
                lambda n, size: self.__get_parameters_network_weight_init(n, size),
            AttributeToTune.NEURON_ACTIVATION_FUNCTION:
                lambda n, size: self.__get_parameters_neuron_activation_function(n, size),
            AttributeToTune.DROPOUT_REGULARIZATION:
                lambda n, size: self.__get_parameters_dropout_regularization(n, size),
            AttributeToTune.NUMBER_NEURONS:
                lambda n, size: self.__get_parameters_number_neurons(n, size)
        }
