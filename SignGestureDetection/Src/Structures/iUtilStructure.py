import os
import json
from enum import Enum
from path import MODEL_PICKELS_FILE
from Exception.structureException import StructureException, StructureFileElementDoesNotExists
from Exception.inputOutputException import PathDoesNotExistException


class IUtilStructure(object):

    @staticmethod
    def save_pickels_used(structure, pickels_name, model_name):
        data = {}

        if not isinstance(structure, Structure):
            raise StructureException("Structure selected is not a valid one")

        if os.path.exists(MODEL_PICKELS_FILE):
            with open(MODEL_PICKELS_FILE) as file:
                data = json.load(file)
                file.close()

        data[structure.value][model_name] = pickels_name

        with open(MODEL_PICKELS_FILE, 'w') as file:
            json.dump(data, file)
            file.close()

    @staticmethod
    def get_pickels_used(structure, model_name):
        if not isinstance(structure, Structure):
            raise StructureException("Structure selected is not a valid one")

        if not os.path.exists(MODEL_PICKELS_FILE):
            raise PathDoesNotExistException("File " + MODEL_PICKELS_FILE + "not found")

        with open(MODEL_PICKELS_FILE) as file:
            data = json.load(file)
            file.close()

        if structure.value not in data or model_name not in data[structure.value]:
            raise StructureFileElementDoesNotExists("There is no " + structure.value + " with the " + model_name +
                                                    " model")
        pickels = data[structure.value][model_name]
        return pickels.split("-")


class Structure(Enum):
    DecisionTree = "decision_tree"
    NeuralNetwork = "neural_network"
