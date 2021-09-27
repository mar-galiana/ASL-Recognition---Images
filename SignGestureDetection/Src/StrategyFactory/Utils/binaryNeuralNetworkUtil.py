import numpy as np
from Model.modelEnum import Environment
from Exception.parametersException import IncorrectVariableType
from Structures.NeuralNetworks.neuralNetworkEnum import LabelsRequirement


class BinaryNeuralNetworkUtil:
    """
    A class to execute the common functionalities in the binary neural networks strategies

    Attributes
    ----------
    model : Model
        A class used to sync up all the functionalities that refer to the database

    Methods
    -------
    remove_not_wanted_labels(self, environment, labels_requirement)
        Remove the dataset samples that will not be used in the training and in the prediction
    """
    def __init__(self, model):
        """
        model : Model
            A class used to sync up all the functionalities that refer to the database
        """
        self.model = model

    def remove_not_wanted_labels(self, environment, labels_requirement):
        """Remove the dataset samples that will not be used in the training and in the prediction

        Parameters
        ----------
        environment : Environment
            Types of environments in a dataset
        labels_requirement : LabelsRequirement
            TODO
        
        Raises
        ------
        IncorrectVariableType
            If the environment variable is not an Environment enumeration
        IncorrectVariableType
            If the labels_requirement variable is not an LabelsRequirement enumeration
        """
        if not isinstance(environment, Environment):
            raise IncorrectVariableType("Expected Environment enum variable")

        if not isinstance(labels_requirement, LabelsRequirement):
            raise IncorrectVariableType("Expected LabelsRequirement enum variable")

        if labels_requirement is not LabelsRequirement.ALL:

            y_train = self.model.get_y(environment)
            x_train = self.model.get_x(environment)

            indexes = [i for i, label in enumerate(y_train) if not self.__is_required(label, labels_requirement)]

            y_train = np.delete(y_train, indexes)
            x_train = np.delete(x_train, indexes, axis=0)

            self.model.set_y(environment, y_train)
            self.model.set_x(environment, x_train)

    @staticmethod
    def __is_required(label, labels_requirement):
        is_required = True

        if labels_requirement == LabelsRequirement.NUMERIC:
            is_required = label.isnumeric()

        elif labels_requirement == LabelsRequirement.ALPHA:
            is_required = label.isalpha()

        elif labels_requirement == LabelsRequirement.ABC:
            is_required = label == 'A' or label == 'B' or label == 'C'

        return is_required
