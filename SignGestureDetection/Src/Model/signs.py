import json
from Constraints.path import SIGNS_FILE
from Exception.modelException import SignIsNotInJsonFileException, SignsFileHasNotBeenReadException


class Signs:
    """
    A class used store the different types of signs stored in the database

    Attributes
    ----------
    signs_dict : dictionary
        a formatted string to print out what the animal says

    Methods
    -------
    read_sign_json()
        Save the dataset into a pickle
    get_signs_dictionary(self)
    get_sign_value(self, sign)
    transform_labels_to_sign_values(self, labels)
    get_sign_based_on_value(self, sign_value)
    """

    def __init__(self):
        self.__signs_dict = None

    def __check_sign_is_not_null(self):
        if self.__signs_dict is None:
            self.read_sign_json()

    def read_sign_json(self):
        with open(SIGNS_FILE) as f:
            file_content = json.load(f)
            self.__signs_dict = file_content.get("signs")

    def get_signs_dictionary(self):
        self.__check_sign_is_not_null()

        return self.__signs_dict

    def get_sign_value(self, sign):
        self.__check_sign_is_not_null()

        if sign not in self.__signs_dict:
            raise SignIsNotInJsonFileException("The sign '" + sign + "' is not in the json file")

        return self.__signs_dict[sign]

    def transform_labels_to_sign_values(self, labels):
        self.__check_sign_is_not_null()

        values = []
        for aux in range(len(labels)):
            values.append(self.__signs_dict.get(labels[aux]))

        return values

    def get_sign_based_on_value(self, sign_value):
        if not isinstance(sign_value, int):
            raise SignIsNotInJsonFileException("In the signs json file the values are all numbers, the character '" +
                                               sign_value + "' is not accepted")
                                            
        self.__check_sign_is_not_null()

        signs_list = list(self.__signs_dict.values())

        if sign_value not in signs_list:
            raise SignIsNotInJsonFileException("The number '" + sign_value + "' is not a value in the sign json file")

        index = signs_list.index(sign_value)
        sign = list(self.__signs_dict.keys())[index]
        return sign
