import numpy as np
from xgboost import XGBClassifier
from Model.enumerations import Environment


class DecisionTree:

    def __init__(self, logger, model, decision_tree_util):
        self.logger = logger
        self.model = model
        self.dt_util = decision_tree_util

    def resize_data(self, environment, shape):
        if len(shape) <= 3:
            x_data = self.model.get_x(environment).reshape(shape[0], shape[1]*shape[2])
        else:
            x_data = self.model.get_x(environment).reshape(shape[0], shape[1]*shape[2]*shape[3])
        return x_data

    def train_model(self):
        self.logger.write_info("Start training model")

        x_train = self.resize_data(Environment.TRAIN, self.model.get_x(Environment.TRAIN).shape)
        x_test = self.resize_data(Environment.TEST, self.model.get_x(Environment.TEST).shape)
        y_train = self.model.get_sign_values(self.model.get_y(Environment.TRAIN))
        y_test = self.model.get_sign_values(self.model.get_y(Environment.TEST))

        xgboost_model = XGBClassifier()
        xgboost_model.fit(x_train, y_train, verbose=True, eval_set=[(x_test, y_test)])

        return xgboost_model
