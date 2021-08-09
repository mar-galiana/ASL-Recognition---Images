import os
import pickle
import numpy as np
from xgboost import plot_tree
import matplotlib.pyplot as plt
from Model.enumerations import Environment
from Structures.iUtilStructure import IUtilStructure, Structure
from path import DECISION_TREE_MODEL_PATH, DECISION_TREE_PLOT_PATH
from Exception.inputOutputException import PathDoesNotExistException


class DecisionTreeUtil(IUtilStructure):

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
        model_path, model_name = self.__get_keras_model_path()

        pickle.dump(xgboost_model, open(model_path + model_name, "wb"))

        super(DecisionTreeUtil, self).save_pickels_used(Structure.DecisionTree, self.model.get_pickels_name(),
                                                        model_name)

        self.logger.write_info("A new decision tree model has been created with the name of: " + model_name + "\n"
                               "In the path: " + model_path + "\n"
                               "This is the name that will be needed in the other strategies if you want to work with "
                               "this model.")

    def read_decision_tree_model(self, name_dt_model):
        dt_model_path = DECISION_TREE_MODEL_PATH + name_dt_model

        if not os.path.exists(dt_model_path):
            raise PathDoesNotExistException("The model needs to exists to be able to use it")

        pickels = super(DecisionTreeUtil, self).get_pickels_used(Structure.DecisionTree, name_dt_model)
        self.model.set_pickels_name(pickels)

        xgboost_model = pickle.load(open(dt_model_path, "rb"))

        return xgboost_model

    def __get_keras_model_name_path(self):
        return self.model.get_pickels_name() + "_model"

    def __get_keras_model_path(self):
        file_name = self.__get_keras_model_name_path()
        return DECISION_TREE_MODEL_PATH, file_name + ".pickle.dat"

    def show_decision_tree(self, xgboost_model):
        file_name = self.__get_keras_model_name_path()

        plot_tree(xgboost_model)
        fig = plt.gcf()
        fig.set_size_inches(30, 15)
        fig.savefig(DECISION_TREE_PLOT_PATH + file_name)
        plt.show()
