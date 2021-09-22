import os
import pickle
import numpy as np
from xgboost import plot_tree
import matplotlib.pyplot as plt
from Structures.iUtilStructure import IUtilStructure, Structure
from Constraints.path import DECISION_TREE_MODEL_PATH, DECISION_TREE_PLOT_PATH
from Exception.inputOutputException import PathDoesNotExistException


class DecisionTreeUtil(IUtilStructure):

    def __init__(self, logger, model):
        self.logger = logger
        self.model = model

    def save_model(self, model):
        model_path, model_name = self.__get_keras_model_path()

        pickle.dump(model, open(model_path + model_name, "wb"))

        super(DecisionTreeUtil, self).save_pickles_used(Structure.DecisionTree, self.model.get_pickles_name(),
                                                        model_name)

        self.logger.write_info("A new decision tree model has been created with the name of: " + model_name + "\n"
                               "In the path: " + model_path + "\n"
                               "This is the name that will be needed in the other strategies if you want to work with "
                               "this model.")

    def load_model(self, name_model):
        dt_model_path = DECISION_TREE_MODEL_PATH + name_model

        if not os.path.exists(dt_model_path):
            raise PathDoesNotExistException("The model needs to exists to be able to use it")

        pickles = super(DecisionTreeUtil, self).get_pickles_used(Structure.DecisionTree, name_model)
        self.model.set_pickles_name(pickles)

        xgboost_model = pickle.load(open(dt_model_path, "rb"))

        return xgboost_model

    @staticmethod
    def resize_single_image(image):
        resized_image = image.reshape(1, image.shape[0]*image.shape[1])
        return resized_image

    def __get_keras_model_name_path(self):
        return self.model.get_pickles_name() + "_model"

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
