import os
import gzip
import numpy as np
import _pickle as cPickle
from Constraints.path import PICKLES_PATH
from Model.modelEnum import Environment, Image
from Exception.modelException import EnvironmentException
from Exception.inputOutputException import PathDoesNotExistException


class InputModel:
    """
    A class used to read the samples stored in the pickles' files.

    Attributes
    ----------
    NUMBER_IMAGES_IN_REDUCED_PICKLE : number
        Number of samples to read for each sign for each pickle
    __train_data : dictionary
        Dictionary of training data and labels
    __test_data : dictionary
        Dictionary of testing data and labels
    pickles_name : array
        Array of pickles' name
    base_pickle_src : string
        Sources of the pickles' path

    Methods
    -------
    set_x(environment, data)
        Set the data
    set_y(environment, labels)
        Set the labels
    set_pickles_name(pickles_name)
        Set the pickles' name
    get_pickles_name()
        Get the pickles' name
    get_data(environment)
        Get the environment's data
    combine_pickles_reducing_size(environment)
        Read a part of each pickle and combine them 
    """

    NUMBER_IMAGES_IN_REDUCED_PICKLE = 5

    def __init__(self):
        self.__train_data = None
        self.__test_data = None
        self.pickles_name = []
        self.base_pickle_src = f"{PICKLES_PATH}%s/%s_%s.pkl"

    def set_x(self, environment, data):
        """Set the data.

        Parameters
        ----------
        environment : Environment
            Environment of the data to set
        data : array
            Data to be assigned to
        """
        self.__set_value(environment, Image.DATA, data)

    def set_y(self, environment, labels):
        """Set the labels.

        Parameters
        ----------
        environment : Environment
            Environment of the labels to set
        labels : array
            Labels to be assigned to
        """
        self.__set_value(environment, Image.LABEL, labels)
    
    def set_pickles_name(self, pickles_name):
        """Set the pickles' name.

        Parameters
        ----------
        pickles_name : array
            Array of pickles to be asigned to
        """
        self.pickles_name = pickles_name

    def get_pickles_name(self):
        """Get the pickles' name.

        Returns
        ----------
        string
            String of the pickles'name separated by a comma
        """
        return "-".join(self.pickles_name)

    def get_data(self, environment):
        """Get the environment's data.

        Raises
        ------
        EnvironmentException
            If the environment variable is not an Environment enumeration

        Returns
        ----------
        dictionary
            Dictionary of the environment data and labels
        """
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if environment == Environment.TRAIN:
            if self.__train_data is None:
                self.__train_data = self.__read_data(environment)
            data = self.__train_data

        else:
            if self.__test_data is None:
                self.__test_data = self.__read_data(environment)
            data = self.__test_data

        return data

    def combine_pickles_reducing_size(self, environment):
        """Read a part of each pickle and combine them.

        Parameters
        ----------
        environment : Environment
            Environment of the pickles to combine

        Raises
        ------
        EnvironmentException
            If the environment variable is not an Environment enumeration
        """
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        data = self.__get_init_data()

        for pickle_name in self.pickles_name:
            actual_pickle_data = self.__get_pickle_data(pickle_name, environment)
            actual_pickle_data = self.__get_firsts_values_each_sign(actual_pickle_data)
            data = self.__concat_pickles(data, actual_pickle_data)

        if environment is Environment.TRAIN:
            self.__train_data = data
        else:
            self.__test_data = data

    def __read_data(self, environment):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        data = self.__get_init_data()

        for pickle_name in self.pickles_name:

            actual_pickle_data = self.__get_pickle_data(pickle_name, environment)

            data = self.__concat_pickles(data, actual_pickle_data)

        return data

    def __get_firsts_values_each_sign(self, actual_pickle_data):
        data = self.__get_init_data()
        data[Image.DESCRIPTION.value] = actual_pickle_data[Image.DESCRIPTION.value]

        signs = np.unique(actual_pickle_data[Image.LABEL.value])

        for sign in signs:
            indexes = [index for index, value in enumerate(actual_pickle_data[Image.LABEL.value]) if value == sign][:self.NUMBER_IMAGES_IN_REDUCED_PICKLE]

            if len(data[Image.DATA.value]) == 0:
                data[Image.DATA.value] = actual_pickle_data[Image.DATA.value][indexes]
                data[Image.LABEL.value] = actual_pickle_data[Image.LABEL.value][indexes]
            else:
                data[Image.DATA.value] = np.concatenate((
                    data[Image.DATA.value],
                    actual_pickle_data[Image.DATA.value][indexes]
                ))
                data[Image.LABEL.value] = np.concatenate((
                    data[Image.LABEL.value],
                    actual_pickle_data[Image.LABEL.value][indexes]
                ))

        return data
    
    def __set_value(self, environment, data_type, values):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if self.__train_data is None:
            self.__train_data = {}

        if environment == Environment.TRAIN:
            self.__train_data[data_type.value] = values
        else:
            self.__test_data[data_type.value] = values

    def __get_pickle_data(self, pickle_name, environment):
        pickle_src = self.base_pickle_src % (pickle_name, pickle_name, environment.value)

        if not os.path.exists(pickle_src):
            raise PathDoesNotExistException("The pickle needs to exists before using it")

        with gzip.open(pickle_src, 'rb') as f:
            actual_pickle_data = cPickle.load(f)
        
        actual_pickle_data[Image.DATA.value] = np.array(actual_pickle_data[Image.DATA.value])
        actual_pickle_data[Image.LABEL.value] = np.array(actual_pickle_data[Image.LABEL.value])

        return actual_pickle_data

    @staticmethod
    def __get_init_data():
        return {
            Image.DESCRIPTION.value: "",
            Image.DATA.value: np.array([]),
            Image.LABEL.value: np.array([])
        }
    
    @staticmethod
    def __concat_pickles(data, actual_pickle_data):
        if len(data[Image.DATA.value]) == 0:
            data[Image.DATA.value] = actual_pickle_data[Image.DATA.value]
            data[Image.LABEL.value] = actual_pickle_data[Image.LABEL.value]
            data[Image.DESCRIPTION.value] = actual_pickle_data[Image.DESCRIPTION.value]

        else:
            data[Image.DATA.value] = np.concatenate((
                data[Image.DATA.value],
                actual_pickle_data[Image.DATA.value]
            ))
            data[Image.LABEL.value] = np.concatenate((
                data[Image.LABEL.value],
                actual_pickle_data[Image.LABEL.value]
            ))
            data[Image.DESCRIPTION.value] += "; " + actual_pickle_data[Image.DESCRIPTION.value]

        return data
