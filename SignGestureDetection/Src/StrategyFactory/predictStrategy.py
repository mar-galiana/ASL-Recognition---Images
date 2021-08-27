import os
from path import IMAGES_PATH
from StrategyFactory.iStrategy import IStrategy
from Structures.iUtilStructure import Structure
from Exception.inputOutputException import InputException
from Structures.DecisionTree.decisionTree import DecisionTree
from Structures.NeuralNetworks.neuralNetwork import NeuralNetwork
from Structures.NeuralNetworks.enumerations import NeuralNetworkEnum
from Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class PredictStrategy(IStrategy):

    def __init__(self, logger, model, nn_util, dt_util, arguments):
        self.logger = logger
        self.model = model
        self.__show_arguments_entered(arguments)

        if arguments[0] not in Structure._value2member_map_:
            raise InputException(arguments[0] + " is not a valid structure")

        self.structure_util = (nn_util, dt_util)[arguments[0] is Structure.NeuralNetwork]
        self.type_structure = Structure(arguments[0])
        self.name_model = arguments[1]
        self.image_name = arguments[2]

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Structure: " + arguments[0] + "\n" \
                         "\t* Model file: " + arguments[1] + "\n" \
                         "\t* Image path: " + arguments[2]
        self.logger.write_info(info_arguments)

    def execute(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        image = self.model.load_image(IMAGES_PATH + self.image_name, True)
        structure_model, resized_image = self.get_model(image)
        prediction = structure_model.predict(resized_image)

        self.logger.write_info("Strategy executed successfully")

    def get_model(self, image):
        if self.type_structure is Structure.NeuralNetwork:
            structure_model, nn_type = self.structure_util.load_model(self.name_model)
            resized_image = self.structure_util.resize_single_image(image, nn_type)
        else:
            structure_model = self.structure_util.load_model(self.name_model)
            resized_image = self.structure_util.resize_single_image(image)

        return structure_model, resized_image
