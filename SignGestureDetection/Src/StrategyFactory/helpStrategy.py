from Model.enumerations import Dataset
from StrategyFactory.iStrategy import IStrategy
from path import NEURAL_NETWORK_MODEL_PATH, DECISION_TREE_MODEL_PATH


class HelpStrategy(IStrategy):

    def __init__(self, logger):
        self.logger = logger
        self.datasets = "\n\t\t* ".join(list(map(lambda c: c.value, Dataset)))

    def execute(self):
        information = "This project contains three different strategies:\n" + \
                      self.__get_information_save_database_strategy() + \
                      self.__get_information_train_nn_strategy() + \
                      self.__get_information_accuracy_nn_strategy() + \
                      self.__get_information_train_decision_tree_strategy() + \
                      self.__get_information_accuracy_dt_strategy() + \
                      self.__get_information_help_strategy()

        self.logger.write_message(information)

    def __get_information_save_database_strategy(self):
        return "* Save Database:\n" \
               "\tThis strategy will save all the images, from the directory selected inside the " \
               "directory: Assets/Dataset/Images, in two pickles (test and train). To execute it "\
               "you need the following arguments:\n " \
               "\t\t--saveDatabase <string> <string> <boolean> <boolean>\n" \
               "\t· The first string has to contain the name of the pickel to use, for example:\n" \
               "\t\t\tsign_gesture_gray_150x150px\n" \
               "\t· The second string indicates witch model to use, the possibilities are:\n" \
               "\t\t* " + self.datasets + "\n" \
               "\t· The first boolean will be true if the directory GestureImageData contains the " \
               "images separated in test and train.\n" \
               "\t· The second boolean will be true if the dataset has to be saved in gray " \
               "colors.\n\n"

    def __get_information_train_nn_strategy(self):
        return "* Train Neural Network:\n"\
               "\tThis strategy will train an specific neural network based on the models stored "\
               "in the Dataset/Pickels. To execute it you need the following arguments:\n "\
               "\t\t--trainNeuralNetwork <string> <string> ...\n"\
               "\t· The first string specifies the type of Neural Network to train, the possibilities are:\n"\
               "\t\t* cnn: Convolutional Neural Network.\n" + \
               self.__get_information_to_select_pickel("other", "--trainNeuralNetwork nn")

    def __get_information_accuracy_nn_strategy(self):
        example_model_name = "cnn_sign_gesture_optimizer_150x150px_model.h5"
        example_execution = "--accuracyNeuralNetwork cnn " + example_model_name

        return "* Accuracy Neural Network:\n"\
               "\tThis strategy will show the accuracy of the Neural Network selected. In order to be able to do it, " \
               "it will need to execute the Train Neural Network before, so a model is stored in the " \
               "Assets/NeuralNetworkModel directory. To execute it you need the following arguments:\n "\
               "\t\t--accuracyNeuralNetwork <string>\n" + \
               self.__get_information_to_select_model(NEURAL_NETWORK_MODEL_PATH, example_model_name, example_execution)

    def __get_information_train_decision_tree_strategy(self):
        return "* Train Decision Tree:\n" \
               "\tThis strategy will train a decision tree based on the models stored in the Dataset/Pickels. To " \
               "execute it you need the following arguments:\n" \
               "\t\t--trainDecisionTree <string> ...\n" + \
               self.__get_information_to_select_pickel("", "--decisionTree")

    def __get_information_accuracy_dt_strategy(self):
        example_model_name = "sign_gesture_gray_150x150px_model.pickle.dat"
        example_execution = "--accuracyDecisionTree " + example_model_name

        return "* Accuracy Decision Tree:\n" \
               "\tThis strategy will show the accuracy of the decision tree selected. In order to be able to do it, " \
               "it will need to execute the Decision Tree Strategy before, so a model is stored in the " \
               "Assets/DecisionTreeModel directory. To execute it you need the following arguments:\n " \
               "\t\t--accuracyDecisionTree <string>\n" + \
               self.__get_information_to_select_model(DECISION_TREE_MODEL_PATH, example_model_name, example_execution)

    @staticmethod
    def __get_information_predict_strategy():
        return "--predict neural_network cnn_asl_alphabet_gray_150x150px-sign_gesture_gray_150x150px_model.h5 " \
               "AslAlphabet/3/3_test.jpg "

    @staticmethod
    def __get_information_help_strategy():
        return "* Help:\n"\
               "\tThis strategy will show all the needed arguments information in order to run this project. It needs" \
               " the argument:\n "\
               "\t\t--help\n\n"

    @staticmethod
    def __get_information_to_select_pickel(argument_position, strategy):
        return "\t· The " + argument_position + " strings have to contain the name of the directories storing the " \
               "pickels. For example, if the pickel you want to use is in the path:\n " \
               "\t\t\tAssets/Dataset/Pickels/sign_gesture_gray_150x150px" \
               "/sign_gesture_gray_150x150px_train.pkl\n " \
               "\t  You will have to enter:\n" \
               "\t\t\tsign_gesture_gray_150x150px\n" \
               "\t If you want to use more than one pickel you can do it by adding all their names in the following " \
               "format:\n" \
               "\t\t\t" + strategy + " sign_gesture_gray_150x150px sign_gesture_gray_150x150px " \
               "sign_gesture_optimizer_150x150px\n\n "

    @staticmethod
    def __get_information_to_select_model(model_directory_path, model_name, example_execution):
        return "\t· The string will contain the name of the file storing the neural network " \
               "model. This model will be in the " + model_directory_path + "path. Once the train strategy has been " \
               "executed, the name of the new model will be displayed in the console, this will be the " + \
               argument_position + " string required.\n" \
               "\tFor example, if the model is in the following path: " + model_directory_path + model_name + ", the " \
               "arguments needed in this strategy will be:\n\t\t" + example_execution + "\n\n"
