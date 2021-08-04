import numpy as np
from StrategyFactory.iStrategy import IStrategy
from DecisionTree.decisionTree import DecisionTree


class DecisionTreeStrategy(IStrategy):

    def __init__(self, logger, model, dt_util, model_util):
        self.logger = logger
        self.model = model
        self.dt_util = dt_util
        self.model_util = model_util
        self.decisionTree = DecisionTree(self.logger, self.model, self.dt_util)

    def execute(self):

        xgboost_model = self.decisionTree.train_model()
        self.logger.write_info("Finished training the decision tree")
        self.dt_util.save_decision_tree_model(xgboost_model)
        self.logger.write_info("Strategy executed successfully")
