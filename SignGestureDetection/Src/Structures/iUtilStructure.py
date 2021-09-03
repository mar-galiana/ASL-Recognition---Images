import os
import json
from enum import Enum
from abc import abstractmethod
from Constraints.path import MODEL_PICKELS_FILE
from Exception.structureException import LabelsRequirementException
from Exception.inputOutputException import PathDoesNotExistException
from Structures.NeuralNetworks.neuralNetworkEnum import LabelsRequirement
from Exception.structureException import StructureException, StructureFileElementDoesNotExists


class IUtilStructure(object):

    @abstractmethod
    def save_model(self, model):
        pass

    @abstractmethod
    def load_model(self, name_model):
        pass

    @abstractmethod
    def resize_single_image(self, image):
        pass

    def save_pickels_used(self, structure, pickels_name, model_name, restriction=LabelsRequirement.ALL):

        if not isinstance(structure, Structure):
            raise StructureException("Structure selected is not a valid one")

        if not isinstance(restriction, LabelsRequirement):
            raise LabelsRequirementException("Label requirement selected is not a valid one")

        if os.path.exists(MODEL_PICKELS_FILE):
            with open(MODEL_PICKELS_FILE) as file:
                data = json.load(file)

        data = self.__get_new_data_structure(data, structure, pickels_name, model_name, restriction)

        with open(MODEL_PICKELS_FILE, 'w') as file:
            json.dump(data, file)

    def get_pickels_used(self, structure, model_name):

        if not isinstance(structure, Structure):
            raise StructureException("Structure selected is not a valid one")

        if not os.path.exists(MODEL_PICKELS_FILE):
            raise PathDoesNotExistException("File " + MODEL_PICKELS_FILE + "not found")

        with open(MODEL_PICKELS_FILE) as file:
            data = json.load(file)

        if structure.value not in data or model_name not in data[structure.value]:
            raise StructureFileElementDoesNotExists("There is no " + structure.value + " with the " + model_name +
                                                    " model")

        return self.__read_data(data, structure, model_name)

    @staticmethod
    def __get_new_data_structure(data, structure, pickels_name, model_name, restriction):
        if structure.value not in data:
            data[structure.value] = {}

        if structure is Structure.CategoricalNeuralNetwork:
            data[structure.value][model_name] = {
                NeuralNetworkInformation.Pickel.value: pickels_name,
                NeuralNetworkInformation.Type.value: model_name.split("_")[0]
            }

        elif structure is Structure.BinaryNeuralNetwork:
            data[structure.value][model_name] = {
                NeuralNetworkInformation.Pickel.value: pickels_name,
                NeuralNetworkInformation.Restriction.value: restriction.value
            }

        else:
            data[structure.value][model_name] = pickels_name

        return data

    @staticmethod
    def __read_data(data, structure, model_name):
        if structure is Structure.CategoricalNeuralNetwork:
            neural_network = data[structure.value][model_name]
            pickels = neural_network[NeuralNetworkInformation.Pickel.value].split("-")
            nn_type = neural_network[NeuralNetworkInformation.Type.value]
            values = pickels, nn_type

        elif structure is Structure.BinaryNeuralNetwork:
            neural_network = data[structure.value][model_name]
            pickels = neural_network[NeuralNetworkInformation.Pickel.value]
            values = pickels.split("-")

        else:
            pickels = data[structure.value][model_name]
            values = pickels.split("-")

        return values


class Structure(Enum):
    DecisionTree = "decision_tree"
    CategoricalNeuralNetwork = "categorical_neural_network"
    BinaryNeuralNetwork = "binary_neural_network"


class NeuralNetworkInformation(Enum):
    Pickel = "pickel"
    Type = "type"
    Restriction = "restriction"
