from xgboost import XGBClassifier
from Model.modelEnum import Environment
from Structures.iUtilStructure import Structure


class DecisionTree:
    """
    A class that contains the decision tree structure and all its functinalities.

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    dt_util : DecisionTreeUtil
        A class to execute the common functionalities of a decision tree structure

    Methods
    -------
    train_model()
        Trains the deicision tree model based on the training samples of the database
    """
    def __init__(self, logger, model, decision_tree_util):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        dt_util : DecisionTreeUtil
            A class to execute the common functionalities of a decision tree structure
        """
        self.logger = logger
        self.model = model
        self.dt_util = decision_tree_util

    def train_model(self):
        """Trains the deicision tree model based on the training samples of the database

        Returns
        -------
        XGBClassifier
            The decision tree model trained
        """
        self.logger.write_info("Start training model")

        self.model.resize_data(Structure.DecisionTree, Environment.TRAIN)

        x_train = self.model.get_x(Environment.TRAIN)
        y_train = self.model.get_signs_values(self.model.get_y(Environment.TRAIN))

        xgboost_model = XGBClassifier()
        xgboost_model.fit(x_train, y_train, verbose=True)

        return xgboost_model
