import os
import numpy as np
from sklearn.metrics import accuracy_score
from Model.enumerations import Environment
from StrategyFactory.iStrategy import IStrategy
from DecisionTree.decisionTree import DecisionTree


class AccuracyDecisionTreeStrategy(IStrategy):

    def __init__(self, logger, model, dt_util, arguments):
        self.logger = logger
        self.model = model
        self.dt_util = dt_util
        self.__show_arguments_entered(arguments)
        self.name_dt_model = arguments[0]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Decision Tree model file: " + arguments[0]
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        decision_tree, xgboost_model = self.__get_decision_tree_model()
        self.__perform_test_data(decision_tree, xgboost_model)
        decision_tree.show_decision_tree(xgboost_model)
        self.logger.write_info("Strategy executed successfully")

    def __get_decision_tree_model(self):

        decision_tree = DecisionTree(self.logger, self.model, self.dt_util)
        xgboost_model = self.dt_util.read_decision_tree_model(self.name_dt_model)
        return decision_tree, xgboost_model

    @staticmethod
    def __get_accuracy(y_pred, y_values):
        # Converting predictions to label
        prediction = list()
        for i in range(len(y_pred)):
            prediction.append(np.argmax(y_pred[i]))

        # Converting one hot encoded test label to label
        values = list()
        for i in range(len(y_values)):
            values.append(np.argmax(y_values[i]))

        accuracy = accuracy_score(prediction, values)
        return accuracy*100

    def __perform_test_data(self, decision_tree, xgboost_model):
        labels_dict = self.dt_util.get_labels_dictionary()
        shape = self.model.get_x(Environment.TEST).shape

        x_test = decision_tree.resize_data(Environment.TEST, shape)
        y_test = self.dt_util.convert_labels_to_numbers(labels_dict, self.model.get_y(Environment.TEST))
        y_pred = xgboost_model.predict(x_test)

        accuracy = self.__get_accuracy(y_pred, y_test)
        self.logger.write_info("Accuracy is: " + "{:.2f}".format(accuracy) + "%")
