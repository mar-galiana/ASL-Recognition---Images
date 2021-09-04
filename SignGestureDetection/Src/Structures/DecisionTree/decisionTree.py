import numpy as np
from xgboost import XGBClassifier
from Model.modelEnum import Environment
from Structures.iUtilStructure import Structure


class DecisionTree:

    def __init__(self, logger, model, decision_tree_util):
        self.logger = logger
        self.model = model
        self.dt_util = decision_tree_util

    def train_model(self):
        self.logger.write_info("Start training model")

        self.model.resize_data(Structure.DecisionTree, Environment.TRAIN, self.model.get_x(Environment.TRAIN).shape)

        x_train = self.model.get_x(Environment.TRAIN)
        y_train = self.model.get_sign_values(self.model.get_y(Environment.TRAIN))

        xgboost_model = XGBClassifier()
        xgboost_model.fit(x_train, y_train, verbose=True)

        return xgboost_model
