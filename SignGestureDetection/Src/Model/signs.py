import json
from Constraints.path import SIGNS_FILE
from Exception.parametersException import IncorrectVariableType
from Exception.modelException import SignIsNotInJsonFileException

class Signs:
    """
    A class used to store the different types of signs in the database

    Attributes
    ----------
    signs_dict : dictionary
        Dictionary storing all the signs of the database 

    Methods
    -------
    get_signs_dictionary()
        Return the dictionary of signs
    get_sign_value(sign)
        Return the value based on its sign
    get_signs_based_on_values(values)
        Return the keys based on theis values
    get_sign_based_on_value(sign_value)
        Return the key based on its value
    """

    def __init__(self):
        self.__signs_dict = None

    def get_signs_dictionary(self):
        """Return the dictionary of signs.

        Returns
        -------
        dictionary
            Dictionary storing all the signs of the database 
        """
        self.__check_sign_is_not_null()

        return self.__signs_dict

    def get_sign_value(self, sign):
        """Return the value based on its sign.

        Parameters
        ----------
        sign : string
            Sign to get the value from
        
        Raises
        ------
        SignIsNotInJsonFileException
            If the sign is not found in the dictionary of signs

        Returns
        -------
        number
            Return the sign's value
        """
        self.__check_sign_is_not_null()

        if sign not in self.__signs_dict:
            raise SignIsNotInJsonFileException("The sign '" + sign + "' is not in the json file")

        return self.__signs_dict[sign]

    def get_signs_based_on_values(self, values):
        """Return the keys based on theis values.

        Parameters
        ----------
        values : array
            Array of values to get the keys from

        Returns
        -------
        array
            Return the keys
        """
        self.__check_sign_is_not_null()

        keys = []
        for aux in range(len(values)):
            keys.append(self.__signs_dict.get(values[aux]))

        return keys

    def get_sign_based_on_value(self, value):
        """Return the key based on its value.

        Parameters
        ----------
        value : number
            Value to get the key from

        Raises
        ------
        IncorrectVariableType
            If the value is not an integer variable
        SignIsNotInJsonFileException
            If the sign is not found in the dictionary of signs

        Returns
        -------
        string
            Return the value's key
        """
        if not isinstance(value, int):
            raise IncorrectVariableType("In the signs json file the values are all numbers, the character '" +
                                               value + "' is not accepted")
                                            
        self.__check_sign_is_not_null()

        signs_list = list(self.__signs_dict.values())

        if value not in signs_list:
            raise SignIsNotInJsonFileException("The number '" + value + "' is not a value in the sign json file")

        index = signs_list.index(value)
        key = list(self.__signs_dict.keys())[index]
        return key

    def __check_sign_is_not_null(self):
        if self.__signs_dict is None:
            self.__read_sign_json()

    def __read_sign_json(self):
        with open(SIGNS_FILE) as f:
            file_content = json.load(f)
            self.__signs_dict = file_content.get("signs")