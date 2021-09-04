import numpy as np
from Model.modelEnum import Environment
from sklearn.metrics import accuracy_score
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum


class AccuracyUtil:

    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    @staticmethod
    def get_accuracy(y_pred, y_values):
        # Converting predictions to label
        prediction = list()
        for i in range(len(y_pred)):
            prediction.append(np.argmax(y_pred[i]))

        # Converting one hot encoded test label to label
        values = list()
        for i in range(len(y_values)):
            values.append(np.argmax(y_values[i]))

        accuracy = accuracy_score(prediction, values)
        return accuracy*100

    def perform_test_data(self, structure, structure_model, nn_type=NeuralNetworkTypeEnum.CNN):

        n_classes = np.unique(self.model.get_y(Environment.TEST)).shape[0] + 1
        shape = self.model.get_x(Environment.TEST).shape

        self.model.resize_data(structure, Environment.TEST, shape, nn_type=nn_type)
        x_test = self.model.get_x(Environment.TEST)

        labels = self.model.get_sign_values(self.model.get_y(Environment.TEST))
        y_test = self.model.get_categorical_vectors(Environment.TEST, n_classes)

        y_pred = structure_model.predict(x_test)

        accuracy = self.get_accuracy(y_pred, y_test)
        self.logger.write_info("Accuracy is: " + "{:.2f}".format(accuracy) + "%")

    def __perform_test_data(self, decision_tree, xgboost_model):
        labels_dict = self.model.get_categorical_vectors()
        shape = self.model.get_x(Environment.TEST).shape

        x_test = decision_tree.resize_data(Environment.TEST, shape)
        y_test = self.dt_util.convert_labels_to_numbers(labels_dict, self.model.get_y(Environment.TEST))
        y_pred = xgboost_model.predict(x_test)

        accuracy = staticmethod(AccuracyUtil.get_accuracy(y_pred, y_test))
        self.logger.write_info("Accuracy is: " + "{:.2f}".format(accuracy) + "%")
