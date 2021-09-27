import os
import pickle
import numpy as np
from xgboost import plot_tree
import matplotlib.pyplot as plt
from Structures.iUtilStructure import IUtilStructure, Structure
from Constraints.path import DECISION_TREE_MODEL_PATH, DECISION_TREE_PLOT_PATH
from Exception.inputOutputException import PathDoesNotExistException


class DecisionTreeUtil(IUtilStructure):
    """
    A class to execute the common functionalities of a decision tree structure.

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database

    Methods
    -------
    save_model(model)
        Store the decision tree trained model into a file with extension "pickle.dat"
    load_model(name_model)
        Load the decision tree trained model and set the dataset used while trianing it 
    show_decision_tree(xgboost_model)
        Plot the decision tree once trained
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

    def save_model(self, model):
        """Store the decision tree trained model into a file with extension "pickle.dat"

        Parameters
        ----------
        model : Model
            A class used to sync up all the functionalities that refer to the database
        """
        model_path, model_name = self.__get_keras_model_path()

        pickle.dump(model, open(model_path + model_name, "wb"))

        super(DecisionTreeUtil, self).save_pickles_used(Structure.DecisionTree, self.model.get_pickles_name(),
                                                        model_name)

        self.logger.write_info("A new decision tree model has been created with the name of: " + model_name + "\n"
                               "In the path: " + model_path + "\n"
                               "This is the name that will be needed in the other strategies if you want to work with "
                               "this model.")

    def load_model(self, name_model):
        """Load the decision tree trained model and set the dataset used while trianing it

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
        XGBClassifier
            The deciosion tree model
        """
        dt_model_path = DECISION_TREE_MODEL_PATH + name_model

        if not os.path.exists(dt_model_path):
            raise PathDoesNotExistException("The model needs to exists to be able to use it")

        pickles = super(DecisionTreeUtil, self).get_pickles_used(Structure.DecisionTree, name_model)
        self.model.set_pickles_name(pickles)

        xgboost_model = pickle.load(open(dt_model_path, "rb"))

        return xgboost_model
    
    def show_decision_tree(self, xgboost_model):
        """Plot the decision tree once trained

        Parameters
        ----------
        xgboost_model : XGBClassifier
            The deciosion tree model
        """
        file_name = self.__get_keras_model_name_path()

        plot_tree(xgboost_model)
        fig = plt.gcf()
        fig.set_size_inches(30, 15)
        fig.savefig(DECISION_TREE_PLOT_PATH + file_name)
        plt.show()


    def __get_keras_model_name_path(self):
        return self.model.get_pickles_name() + "_model"

    def __get_keras_model_path(self):
        file_name = self.__get_keras_model_name_path()
        return DECISION_TREE_MODEL_PATH, file_name + ".pickle.dat"