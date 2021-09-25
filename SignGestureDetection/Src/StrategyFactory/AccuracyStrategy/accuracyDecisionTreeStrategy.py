import os
from StrategyFactory.iStrategy import IStrategy
from Structures.iUtilStructure import Structure
from Structures.DecisionTree.decisionTree import DecisionTree


class AccuracyDecisionTreeStrategy(IStrategy):

    def __init__(self, logger, model, dt_util, accuracy_util, arguments):
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
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        xgboost_model = self.__get_decision_tree_model()
        self.accuracy_util.perform_test_data(Structure.DecisionTree, xgboost_model)
        self.dt_util.show_decision_tree(xgboost_model)
        self.logger.write_info("Strategy executed successfully")

    def __get_decision_tree_model(self):
        xgboost_model = self.dt_util.load_model(self.name_dt_model)
        return xgboost_model
