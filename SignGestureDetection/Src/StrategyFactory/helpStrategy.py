from Logger.logger import Color
from Model.modelEnum import Dataset
from Structures.iUtilStructure import Structure
from StrategyFactory.iStrategy import IStrategy
from StrategyFactory.strategyEnum import Strategies
from Constraints.path import CATEGORICAL_NEURAL_NETWORK_MODEL_PATH, DECISION_TREE_MODEL_PATH, \
                             BINARY_NEURAL_NETWORK_MODEL_PATH
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum, LabelsRequirement, AttributeToTuneEnum


class HelpStrategy(IStrategy):
    """
    A class to print the information of each strategy

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information.
    datasets : string
        Names of the dataset to use.
    
    Methods
    -------
    execute()
        Show the arguments needed to execute each strategy
    """

    def __init__(self, logger):
        """
        Parameters
        ----------
        logger : Logger
            A class used to show the execution information.
        """
        self.logger = logger
        self.datasets = "\n\t* ".join(list(map(lambda c: c.value, Dataset)))
        self.attributes_to_optimize = "\n\t* ".join(list(map(lambda c: c.value, AttributeToTuneEnum)))
        self.structures = "\n\t* ".join(list(map(lambda c: c.value, Structure)))

    def execute(self):
        """Show the arguments needed to execute each strategy
        """

        information = "This project contains three different strategies:\n\n\n" + \
                      self.__get_information_setup_strategy() + "\n" + \
                      self.__get_information_save_database_strategy() + "\n" + \
                      self.__get_information_train_categorical_nn_strategy() + "\n" + \
                      self.__get_information_accuracy_categorical_nn_strategy() + "\n" + \
                      self.__get_information_train_decision_tree_strategy() + "\n" + \
                      self.__get_information_accuracy_dt_strategy() + "\n" + \
                      self.__get_information_train_binary_nn_strategy() + "\n" + \
                      self.__get_information_accuracy_binary_nn_strategy() + "\n" + \
                      self.__get_information_hyperparameter_optimization_strategy() + "\n" + \
                      self.__get_information_predict_strategy() + "\n" + \
                      self.__get_information_help_strategy()

        self.logger.write_message(information)

    @staticmethod
    def __get_information_setup_strategy():

        return "* " + Color.BOLD.value + "Setup" + Color.END.value + "\n\n" + \
               "This strategy will create the folders and the files needed to execute all the other strategies." \
               "This strategy needs to be the first one to be executed.\n\n" \
               "To execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.SETUP.value + "\n\n" + Color.END.value

    def __get_information_save_database_strategy(self):
        return "* " + Color.BOLD.value + "Save Database" + Color.END.value + "\n\n" \
               "This strategy will save all the images, from the dataset selected, into two pickles: one for " \
               "testing and the other for training. The datasets are stored in the path Assets/Dataset/Images.\n\n" \
               "To execute it you need the following arguments:\n " + \
               Color.BLUE.value + "\t" + Strategies.SAVE_DATABASE.value + " <string> <string> <boolean>\n\n" +\
               Color.END.value +\
               "· The first string has to contain the name of the pickle to use, for example:\n" \
               "\tsign_gesture_gray_150x150px\n\n" \
               "· The second string indicates witch model to use, the possibilities are:\n" \
               "\t* " + self.datasets + "\n\n" \
               "· The boolean will be true if the directory GestureImageData contains the images separated in test " \
               "and train.\n\n"

    def __get_information_train_categorical_nn_strategy(self):
        example_execution = Strategies.TRAIN_CATEGORICAL_NEURAL_NETWORK.value + " " + NeuralNetworkTypeEnum.ANN.value

        return "* " + Color.BOLD.value + "Train Categorical Neural Network" + Color.END.value + "\n\n"\
               "This strategy will train a categorical neural network based on the samples stored "\
               "in the Dataset/Pickles.\n\n" \
               "To execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.TRAIN_CATEGORICAL_NEURAL_NETWORK.value + " <string> <string> " \
               "...\n\n" + Color.END.value +\
               "· The first string specifies the type of Neural Network to train, the possibilities are:\n"\
               "\t* " + NeuralNetworkTypeEnum.ANN.value + ": Artificial Neural Network.\n" + \
               "\t* " + NeuralNetworkTypeEnum.CNN.value + ": Convolutional Neural Network.\n" + \
               "\t* " + NeuralNetworkTypeEnum.IMPROVED_CNN.value + ": Improved Convolutional Neural Network.\n\n" + \
               self.__get_information_to_select_pickle("other", example_execution)

    def __get_information_accuracy_categorical_nn_strategy(self):
        example_model_name = "cnn_sign_gesture_optimizer_150x150px_model.h5"
        example_execution = Strategies.ACCURACY_CATEGORICAL_NEURAL_NETWORK.value + " " + example_model_name

        return "* " + Color.BOLD.value + "Accuracy Categorical Neural Network:" + Color.END.value + "\n\n"\
               "This strategy will show the accuracy of the Neural Network selected. In order to be able to do it, " \
               "you will need to execute the Train Categorical Neural Network before, so a model is stored in the " +\
               CATEGORICAL_NEURAL_NETWORK_MODEL_PATH + " directory.\n\n" \
               "To execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.ACCURACY_CATEGORICAL_NEURAL_NETWORK.value + " <string>\n\n" +\
               Color.END.value +\
               self.__get_information_to_select_model(CATEGORICAL_NEURAL_NETWORK_MODEL_PATH, example_model_name,
                                                      example_execution)

    def __get_information_train_decision_tree_strategy(self):
        return "* " + Color.BOLD.value + "Train Decision Tree" + Color.END.value + "\n\n" \
               "This strategy will train a decision tree based on the models stored in the Dataset/Pickles. To " \
               "execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.TRAIN_DECISION_TREE.value + " <string> ...\n\n" + Color.END.value +\
               self.__get_information_to_select_pickle("", Strategies.TRAIN_DECISION_TREE.value)

    def __get_information_accuracy_dt_strategy(self):
        example_model_name = "sign_gesture_gray_150x150px_model.pickle.dat"
        example_execution = Strategies.ACCURACY_DECISION_TREE.value + " " + example_model_name

        return "* " + Color.BOLD.value + "Accuracy Decision Tree:" + Color.END.value + "\n\n" \
               "This strategy will show the accuracy of the decision tree selected. In order to be able to do it, " \
               "it will need to execute the Decision Tree Strategy before, so a model is stored in the " + \
               DECISION_TREE_MODEL_PATH + " directory. To execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.ACCURACY_DECISION_TREE.value + " <string>\n\n" + Color.END.value + \
               self.__get_information_to_select_model(DECISION_TREE_MODEL_PATH, example_model_name, example_execution)

    def __get_information_train_binary_nn_strategy(self):
        example_execution = Strategies.TRAIN_BINARY_NEURAL_NETWORK.value + " " + LabelsRequirement.ABC.value

        return "* " + Color.BOLD.value + "Train Binary Neural Network" + Color.END.value + "\n\n" \
               "This strategy will train a binary neural network based on the samples stored " \
               "in the Dataset/Pickles.\n\n" \
               "To execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.TRAIN_BINARY_NEURAL_NETWORK.value + " <string> <string> " \
               "...\n\n" + Color.END.value + \
               "· The first string specifies the signs that will be used to train the model, the possibilities are:\n" \
               "\t* " + LabelsRequirement.ABC.value + ": Only the signs 'A', 'B' and 'C'.\n" + \
               "\t* " + LabelsRequirement.NUMERIC.value + ": Only numeric signs.\n" + \
               "\t* " + LabelsRequirement.ALPHA.value + ": Only signs that represents letters.\n" + \
               "\t* " + LabelsRequirement.ALL.value + ": All the signs in the database.\n\n" +\
               self.__get_information_to_select_pickle("other", example_execution)

    def __get_information_accuracy_binary_nn_strategy(self):
        example_model_name = "abc_asl_alphabet_gray_150x150px-sign_gesture_gray_150x150px_models.zip"
        example_execution = Strategies.ACCURACY_BINARY_NEURAL_NETWORK.value + " " + example_model_name

        return "* " + Color.BOLD.value + "Accuracy Binary Neural Network:" + Color.END.value + "\n\n" \
               "This strategy will show the accuracy of the binary neural network selected. In order to be able to do" \
               " it, you will need to execute the Train Binary Neural Network before, so a model is stored in the " +\
               BINARY_NEURAL_NETWORK_MODEL_PATH + " directory.\n\n" \
               "To execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.ACCURACY_BINARY_NEURAL_NETWORK.value + " <string>\n\n" + \
               Color.END.value + \
               self.__get_information_to_select_model(BINARY_NEURAL_NETWORK_MODEL_PATH, example_model_name,
                                                      example_execution)

    def __get_information_hyperparameter_optimization_strategy(self):
        example_execution = Strategies.HYPERPARAMETER_OPTIMIZATION.value + " " + \
                            AttributeToTuneEnum.BATCH_SIZE_AND_EPOCHS.value

        return "* " + Color.BOLD.value + "Hyperparameter Optimization" + Color.END.value + "\n\n" \
               "This strategy will optimize the attributes of the convolutional neural network structure.\n\n" \
               "To execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.HYPERPARAMETER_OPTIMIZATION.value + " <string> <string> " \
               "...\n\n" + Color.END.value + \
               "· The first string specifies the attribute to optimize, the possibilities are:\n" \
               "\t* " + self.attributes_to_optimize + "\n\n" +\
               self.__get_information_to_select_pickle("other", example_execution)

    def __get_information_predict_strategy(self):
        return "* " + Color.BOLD.value + "Predict" + Color.END.value + "\n\n" + \
               "This strategy will predict the value of the input image based on the model selected.\n\n" \
               "To execute it you need the following arguments:\n" + \
               Color.BLUE.value + "\t" + Strategies.PREDICT_IMAGE.value + " " + "<string> <string> <string>\n\n" \
               + Color.END.value +\
               "· The first string specifies the type of model that will be used, the possibilities are:\n" \
               "\t* " + self.structures + "\n\n" + \
               "· The second string will be the name of the file storing the model selected.\n\n" \
               "· The last string will contain the path of the image to predict.\n\n"

    @staticmethod
    def __get_information_help_strategy():
        return "* " + Color.BOLD.value + "Help:" + Color.END.value + "\n"\
               "This strategy will show all the needed arguments information in order to run this project. It needs" \
               " the argument:\n" + \
               Color.BLUE.value + "\t" + Strategies.HELP.value + "\n\n" + Color.END.value

    @staticmethod
    def __get_information_to_select_pickle(argument_position, strategy):
        return "· The " + argument_position + " strings have to contain the name of the directories storing the " \
               "pickles. For example, if the pickle you want to use is in the path: " \
               "Assets/Dataset/Pickles/sign_gesture_gray_150x150px/sign_gesture_gray_150x150px_train.pkl, you will " \
               "have to enter: sign_gesture_gray_150x150px.\n" \
               "If you want to use more than one pickle you can do it by adding all their names in the following " \
               "format:\n" \
               "\t" + strategy + " sign_gesture_gray_150x150px sign_gesture_gray_150x150px \n\n"

    @staticmethod
    def __get_information_to_select_model(model_directory_path, model_name, example_execution):
        return "· The string will contain the name of the file storing the model. This model will be in the " +\
               model_directory_path + " path.\n" \
               "Once the train strategy has been executed, the name of the new model will be displayed in the " \
               "console, this will be the string required.\n" \
               "For example, if the model is in the following path: " + model_directory_path + model_name\
               + ", the arguments needed in this strategy will be:\n" \
               "\t" + example_execution + "\n\n"
