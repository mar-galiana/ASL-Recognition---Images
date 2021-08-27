import numpy as np
from StrategyFactory.iStrategy import IStrategy
from Structures.DecisionTree.decisionTree import DecisionTree


class TrainDecisionTreeStrategy(IStrategy):

    def __init__(self, logger, model, dt_util, arguments):
        self.logger = logger
        self.model = model
        self.dt_util = dt_util
        self.__show_arguments_entered(arguments)

        self.pickels = arguments
        self.decisionTree = DecisionTree(self.logger, self.model, self.dt_util)

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Pickels selected: " + ", ".join(arguments)
        self.logger.write_info(info_arguments)

    def execute(self):
        self.model.set_pickels_name(self.pickels)
        xgboost_model = self.decisionTree.train_model()
        self.logger.write_info("Finished training the decision tree")
        self.dt_util.save_model(xgboost_model)
        self.logger.write_info("Strategy executed successfully")
