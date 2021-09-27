import os
from StrategyFactory.iStrategy import IStrategy
from Structures.iUtilStructure import Structure
from Structures.DecisionTree.decisionTree import DecisionTree


class AccuracyDecisionTreeStrategy(IStrategy):
    """
    A class to predict the value of the image entered

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    dt_util : DecisionTreeUtil
        A class to execute the common functionalities of a decision tree structure
    accuracy_util : AccuracyUtil
        A class to execute the common functionalities in accuracy strategies
    name_dt_model : string
        Name of the file that contains the model to test

    Methods
    -------
    execute()
        Show the accuracy of the decision tree model, previously trained, using the test database
    """
    def __init__(self, logger, model, dt_util, accuracy_util, arguments):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        dt_util : DecisionTreeUtil
            A class to execute the common functionalities of a decision tree structure
        accuracy_util : AccuracyUtil
            A class to execute the common functionalities in accuracy strategies
        arguments : array
            Array of arguments entered in the execution
        """
        self.logger = logger
        self.model = model
        self.dt_util = dt_util
        self.accuracy_util = accuracy_util
        self.__show_arguments_entered(arguments)
        self.name_dt_model = arguments[0]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Decision Tree model file: " + arguments[0]
        self.logger.write_info(info_arguments)

    def execute(self):
        """Show the accuracy of the decision tree model, previously trained, using the test database
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        xgboost_model = self.__get_decision_tree_model()
        self.accuracy_util.perform_test_data(Structure.DecisionTree, xgboost_model)
        self.dt_util.show_decision_tree(xgboost_model)
        self.logger.write_info("Strategy executed successfully")

    def __get_decision_tree_model(self):
        xgboost_model = self.dt_util.load_model(self.name_dt_model)
        return xgboost_model
