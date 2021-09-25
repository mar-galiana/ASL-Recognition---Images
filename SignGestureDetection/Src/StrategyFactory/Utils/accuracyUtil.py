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

        accuracy = accuracy_score(prediction, y_values)
        return accuracy*100

    def perform_test_data(self, structure, structure_model, nn_type=NeuralNetworkTypeEnum.CNN):

        self.model.resize_data(structure, Environment.TEST, nn_type=nn_type)
        x_test = self.model.get_x(Environment.TEST)

        labels = self.model.get_signs_values(self.model.get_y(Environment.TEST))

        y_pred = structure_model.predict(x_test)

        accuracy = self.get_accuracy(y_pred, labels)
        self.logger.write_info("Accuracy is: " + "{:.2f}".format(accuracy) + "%")
