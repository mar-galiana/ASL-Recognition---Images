import pickle
from path import DECISION_TREE_MODEL_PATH


class DecisionTreeUtil:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def convert_labels_to_numbers(labels_dict, labels):
        for aux in range(len(labels)):
            labels[aux] = labels_dict.get(labels[aux])

        return labels

    def save_decision_tree_model(self, xgboost_model):
        file_name = self.__get_keras_model_path()
        pickle.dump(xgboost_model, open(file_name, "wb"))

    def __get_keras_model_path(self):
        file_name = self.model.get_pickel_name() + "_model"
        return DECISION_TREE_MODEL_PATH + file_name + ".pickle.dat"
