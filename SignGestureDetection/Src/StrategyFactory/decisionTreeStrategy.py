from numpy import std, mean
import numpy as np
from Model.enumerations import Environment
from StrategyFactory.iStrategy import IStrategy
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt


class DecisionTreeStrategy(IStrategy):

    def __init__(self, logger, model, model_util):
        self.logger = logger
        self.model = model
        self.model_util = model_util

    def resize_data(self, environment, shape):
        x_data = self.model.get_x(environment).reshape(shape[0], shape[1]*shape[2])
        return x_data

    def execute(self):
        labels_dict = {}
        aux = 0
        for string_y in np.unique(self.model.get_y(Environment.TRAIN)):
            labels_dict[string_y] = aux
            aux += 1
        x_train = self.resize_data(Environment.TRAIN, self.model.get_x(Environment.TRAIN).shape)
        x_test = self.resize_data(Environment.TEST, self.model.get_x(Environment.TEST).shape)
        y_train = self.convert_labels_to_numbers(labels_dict, self.model.get_y(Environment.TRAIN))
        y_test = self.convert_labels_to_numbers(labels_dict, self.model.get_y(Environment.TEST))

        xgboost_model = XGBClassifier()
        xgboost_model.fit(x_train, y_train, verbose=True, eval_set=[(x_test, y_test)])
        plt = plot_tree(xgboost_model, num_trees=1)
        fig = plt.gcf()
        fig.set_size_inches(30, 15)

        pass

    @staticmethod
    def convert_labels_to_numbers(labels_dict, labels):
        for aux in range(len(labels)):
            labels[aux] = labels_dict.get(labels[aux])

        return labels
