import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Constraints.path import SIGNS_IMAGES
from StrategyFactory.iStrategy import IStrategy
from Structures.iUtilStructure import Structure
from Exception.inputOutputException import InputException


class PredictStrategy(IStrategy):
    """
    A class to predict the value of the image entered

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    structure_util : IUtilStructure
        Common functionalities interface for the different util structures
    type_structure : Structure
        Different types of models available to be trained
    name_model : string
        Name of the model used to predict the sign
    image_name : string
        Path of the image used to do the prediction

    Methods
    -------
    execute()
        Predict the image entered based on the model selected
    """

    def __init__(self, logger, model, nn_util, dt_util, arguments):
        """
        Parameters
        ----------
        logger : Logger
            A class used to show the execution information.
        model : Model
            A class used to sync up all the functionalities that refer to the database
        nn_util : NeuralNetworkUtil
            A class to execute the common functionalities of a neural network structure
        dt_util : DecisionTreeUtil
            A class to execute the common functionalities of a decision tree structure
        arguments: array
            Array of arguments entered without the execution strategy
        
         Raises
        ------
        InputException
            If the first argument is not a value of the Structure enumeration
        """
        self.logger = logger
        self.model = model
        self.__show_arguments_entered(arguments)

        if arguments[0] not in Structure._value2member_map_:
            raise InputException(arguments[0] + " is not a valid structure")

        self.structure_util = (nn_util, dt_util)[arguments[0] is Structure.CategoricalNeuralNetwork]
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
        """Predict the image entered based on the model selected
        """

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # image = self.model.load_image(IMAGES_PATH + self.image_name, True)
        image = self.model.load_image(self.image_name, True)

        # normalizing the data to help with the training
        structure_model, resized_image = self.__get_model(image)

        prediction = structure_model.predict(resized_image)
        sign_value = np.int16(np.argmax(prediction)).item()
        sign = self.model.get_sign_based_on_value(sign_value)
        
        self.__show_result(sign)

        self.logger.write_info("Strategy executed successfully")

    def __get_model(self, image):

        if self.type_structure is Structure.CategoricalNeuralNetwork:
            structure_model, nn_type = self.structure_util.load_model(self.name_model)
            resized_image = self.model.resize_image(image, self.type_structure, nn_type=nn_type)
        else:
            structure_model = self.structure_util.load_model(self.name_model)
            resized_image = self.model.resize_image(image, self.type_structure)

        return structure_model, resized_image
    
    def __show_result(self, sign):
        self.logger.write_info("The image represents the sign '" + sign + "'")

        image_path = SIGNS_IMAGES + sign.lower() + ".png"
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
