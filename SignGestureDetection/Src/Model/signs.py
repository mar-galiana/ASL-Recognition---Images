import json
from path import SIGNS_FILE
from Exception.modelException import SignIsNotInJsonFileException


class Signs:

    def __init__(self):
        self.__signs_dict = {}
        self.read_sign_json()

    def read_sign_json(self):
        with open(SIGNS_FILE) as f:
            self.__signs_dict = json.load(f)

    def get_sign_value(self, sign):
        if sign not in self.__signs_dict:
            raise SignIsNotInJsonFileException("The sign '" + sign + "' is not in the json file")

        return self.__signs_dict[sign]

    def transform_labels_to_sign_values(self, labels):
        for aux in range(len(labels)):
            labels[aux] = self.__signs_dict.get(labels[aux])

        return labels

    def get_sign_based_on_value(self, sign_value):
        signs_list = list(self.__signs_dict.values())

        if not isinstance(signs_list, int):
            raise SignIsNotInJsonFileException("In the signs json file the values are all numbers, the character '" +
                                               sign_value + "' is not accepted")

        if sign_value not in signs_list:
            raise SignIsNotInJsonFileException("The number '" + sign_value + "' is not a value in the sign json file")

        return signs_list.index(sign_value)
