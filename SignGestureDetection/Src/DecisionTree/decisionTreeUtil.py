import pickle
import numpy as np
from path import DECISION_TREE_MODEL_PATH
from Model.enumerations import Environment


class DecisionTreeUtil:

    def __init__(self, logger, model):
        self.logger = logger
        self.model = model

    def get_labels_dictionary(self):
        labels_dict = {}
        aux = 0
        for string_y in np.unique(self.model.get_y(Environment.TRAIN)):
            labels_dict[string_y] = aux
            aux += 1
        return labels_dict

    @staticmethod
    def convert_labels_to_numbers(labels_dict, labels):
        for aux in range(len(labels)):
            labels[aux] = labels_dict.get(labels[aux])

        return labels

    def save_decision_tree_model(self, xgboost_model):
        model_name = self.__get_keras_model_path()
        pickle.dump(xgboost_model, open(model_name, "wb"))
        self.logger.write_info("A new decision tree model has been created with the name of: " + model_name + ".\nThis "
                               "is the name that will be needed in the other strategies if you want to work with this "
                               "model.")

    @staticmethod
    def read_decision_tree_model(name_dt_model):
        file_name = DECISION_TREE_MODEL_PATH + name_dt_model
        xgboost_model = pickle.load(open(file_name, "rb"))
        return xgboost_model

    def __get_keras_model_path(self):
        file_name = self.model.get_pickels_name() + "_model"
        return DECISION_TREE_MODEL_PATH + file_name + ".pickle.dat"
