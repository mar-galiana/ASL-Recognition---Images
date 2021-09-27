import os
import json
from enum import Enum
from abc import abstractmethod
from Constraints.path import MODEL_PICKLES_FILE
from Exception.structureException import LabelsRequirementException
from Exception.inputOutputException import PathDoesNotExistException
from Structures.NeuralNetworks.neuralNetworkEnum import LabelsRequirement
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkInformation
from Exception.structureException import StructureException, StructureFileElementDoesNotExists


class IUtilStructure(object):
    """
    Common functionalities interface for the different util structures
    """
    
    @abstractmethod
    def save_model(self, model):
        pass

    @abstractmethod
    def load_model(self, name_model):
        pass

    @abstractmethod
    def resize_single_image(self, image):
        pass

    def save_pickles_used(self, structure, pickles_name, model_name, restriction=LabelsRequirement.ALL):
        """Stores the information used in a model while training it.

        Parameters
        ----------
        structure : Structure
            Different types of models available to be trained
        pickles_name : string
            Name of the pickled used while training the model
        model_name : string
            Model's name
        restriction : LabelsRequirement, optional
            Types of labels allowed to reduce the database size (Default is LabelsRequirement.ALL)
        
        Raises
        ------
        StructureException
            If the structure variable is not an Structure enumeration
        LabelsRequirementException
            If the restriction variable is not an LabelsRequirementException enumeration
        """
        if not isinstance(structure, Structure):
            raise StructureException("Structure selected is not a valid one")

        if not isinstance(restriction, LabelsRequirement):
            raise LabelsRequirementException("Label requirement selected is not a valid one")

        if os.path.exists(MODEL_PICKLES_FILE):
            with open(MODEL_PICKLES_FILE) as file:
                data = json.load(file)

        data = self.__get_new_data_structure(data, structure, pickles_name, model_name, restriction)

        with open(MODEL_PICKLES_FILE, 'w') as file:
            json.dump(data, file)

    def get_pickles_used(self, structure, model_name):
        """Get the information used in a model while training it.

        Parameters
        ----------
        structure : Structure
            Different types of models available to be trained
        model_name : string
            Model's name
        
        Raises
        ------
        StructureException
            If the structure variable is not an Structure enumeration
        PathDoesNotExistException
            If the json storing all the models names does not exist
        StructureFileElementDoesNotExists
            If the model's name is not found in the json file

        Returns
        -------
        If the structure is a categorical neural network it will return two values: 
        array
            Array of pickles used when training the model
        string
            Neural network type, it will be a value of the enumeration NeuralNetworkTypeEnum
        
        If the structure is a binary neural network it will return it will return two value:
        array
            Array of pickles used when training the model
        string
            Samples restriction, it will be a value of the enumeration LabelsRequirement

        If the structure is a set of binary neural networks it will return one value:
        array
            Array of pickles used when training the model

        """

        if not isinstance(structure, Structure):
            raise StructureException("Structure selected is not a valid one")
        
        if not os.path.exists(MODEL_PICKLES_FILE):
            raise PathDoesNotExistException("File " + MODEL_PICKLES_FILE + " not found")

        with open(MODEL_PICKLES_FILE) as file:
            data = json.load(file)

        if structure.value not in data or model_name not in data[structure.value]:
            raise StructureFileElementDoesNotExists("There is no " + structure.value + " with the " + model_name +
                                                    " model")

        return self.__read_data(data, structure, model_name)

    @staticmethod
    def __get_new_data_structure(data, structure, pickles_name, model_name, restriction):
        if structure.value not in data:
            data[structure.value] = {}

        if structure is Structure.CategoricalNeuralNetwork:
            data[structure.value][model_name] = {
                NeuralNetworkInformation.Pickle.value: pickles_name,
                NeuralNetworkInformation.Type.value: model_name.split("_")[0]
            }

        elif structure is Structure.BinaryNeuralNetwork:
            data[structure.value][model_name] = {
                NeuralNetworkInformation.Pickle.value: pickles_name,
                NeuralNetworkInformation.Restriction.value: restriction.value
            }

        else:
            data[structure.value][model_name] = pickles_name

        return data

    @staticmethod
    def __read_data(data, structure, model_name):
        if structure is Structure.CategoricalNeuralNetwork:
            neural_network = data[structure.value][model_name]
            pickles = neural_network[NeuralNetworkInformation.Pickle.value].split("-")
            nn_type = neural_network[NeuralNetworkInformation.Type.value]
            values = pickles, nn_type

        elif structure is Structure.BinaryNeuralNetwork:
            neural_network = data[structure.value][model_name]
            pickles = neural_network[NeuralNetworkInformation.Pickle.value].split("-")
            restriction = neural_network[NeuralNetworkInformation.Restriction.value]
            values = pickles, restriction

        else:
            pickles = data[structure.value][model_name]
            values = pickles.split("-")

        return values


class Structure(Enum):
    """Different types of models available to be trained.
    """
    DecisionTree = "decision_tree"
    CategoricalNeuralNetwork = "categorical_neural_network"
    BinaryNeuralNetwork = "binary_neural_network"
