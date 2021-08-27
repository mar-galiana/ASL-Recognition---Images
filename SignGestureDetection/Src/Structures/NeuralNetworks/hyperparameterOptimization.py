"""
By default, the grid search will only use one thread. By setting the n_jobs argument in the GridSearchCV constructor to
-1, the process will use all cores on your machine

The best_score_ member provides access to the best score observed during the optimization procedure and the best_params_
describes the combination of parameters that achieved the best results.
"""
import numpy
from path import ASSETS_PATH
from Model.enumerations import Environment
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.models import Sequential
from Exception.structureException import IncorrectVariableType
from Structures.NeuralNetworks.enumerations import AttributeToTune
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


def get_sequential_model(num_classes, image_size):
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
    model = get_sequential_model(num_classes, image_size)

    # Compile model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_model_optimizer_algorithm(optimizer='adam', num_classes=39, image_size=None):
    # create model
    model = get_sequential_model(num_classes, image_size)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


class HyperparameterOptimization:

    def __init__(self, logger, model, nn_util):
        self.model = model
        self.logger = logger
        self.nn_util = nn_util
        self.cnn = ConvolutionalNeuralNetwork(logger, model, nn_util)

    def calculate_best_hyperparameter_optimization(self, attribute_tune):

        if not isinstance(attribute_tune, AttributeToTune):
            raise IncorrectVariableType("Expecting AttributeToTune enumeration")

        n_classes, image_size = self.__prepare_data()
        grid = self.__get_grid_search_classifier(attribute_tune, n_classes, image_size)
        grid_result = self.__train_convolutional_neural_network(grid)
        self.__summarize_results(grid_result)

    def __get_grid_search_classifier(self, attribute_tune, n_classes, image_size):
        if attribute_tune is AttributeToTune.BATCH_SIZE_AND_EPOCHS:
            classifier = KerasClassifier(build_fn=create_model_batch_epochs, verbose=2)
            param_grid = self.__get_parameters_for_batch_epochs(n_classes, image_size)
        else:
            classifier = KerasClassifier(build_fn=create_model_optimizer_algorithm, epochs=10, batch_size=10, verbose=2)
            param_grid = self.__get_parameters_for_optimization_algorithm(n_classes, image_size)

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
        return param_grid

    @staticmethod
    def __get_parameters_for_optimization_algorithm(n_classes, image_size):
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(optimizer=optimizer, num_classes=[n_classes], image_size=[image_size])
        return param_grid

    def __summarize_results(self, grid_result):
        self.logger.write_info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for mean, stdev, param in zip(means, stds, params):
            self.logger.write_info("%f (%f) with: %r" % (mean, stdev, param))
