import numpy as np
from Model.modelEnum import Environment
from sklearn.metrics import accuracy_score
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum


class AccuracyUtil:
    """
    A class to execute the common functionalities in accuracy strategies.

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database

    Methods
    -------
    get_accuracy(y_pred, y_values)
        Checks the returned values of a prediction with the correct values and returns the percentage of success
    perform_test_data(structure, structure_model, nn_type=NeuralNetworkTypeEnum.CNN)
        Processes the sample data for training and run the model prediction showing its accuracy
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

    @staticmethod
    def get_accuracy(y_pred, y_values):
        """Checks the returned values of a prediction with the correct values and returns the percentage of success

        Parameters
        ----------
        y_pred : array
            Returned values of the prediction
        y_values : number
            Correct values the prediction is supposed to return

        Returns
        -------
        number
            Percentage of success
        """
        
        prediction = list()
        
        for i in range(len(y_pred)):
            prediction.append(np.argmax(y_pred[i]))

        accuracy = accuracy_score(prediction, y_values)
        return accuracy*100

    def perform_test_data(self, structure, structure_model, nn_type=NeuralNetworkTypeEnum.CNN):
        """Processes the sample data for training and run the model prediction showing its accuracy

        Parameters
        ----------
        structure : Structure
            Different types of models available to be trained
        structure_model : Sequential
            Pre-trained sequential model
        nn_type : NeuralNetworkTypeEnum, optional
            Neural network type (Defult is NeuralNetworkTypeEnum.CNN)
        """

        self.model.resize_data(structure, Environment.TEST, nn_type=nn_type)
        x_test = self.model.get_x(Environment.TEST)

        labels = self.model.get_signs_values(self.model.get_y(Environment.TEST))

        y_pred = structure_model.predict(x_test)

        accuracy = self.get_accuracy(y_pred, labels)
        self.logger.write_info("Accuracy is: " + "{:.2f}".format(accuracy) + "%")
