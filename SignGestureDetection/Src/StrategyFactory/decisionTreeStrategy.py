import pickle
import numpy as np
from xgboost import plot_tree
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from Model.enumerations import Environment
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

        dt_model = self.decisionTree.train_model()
        self.logger.write_info("Finished training the decision tree")
        # TODO save model
