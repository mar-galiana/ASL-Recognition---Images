from StrategyFactory.iStrategy import IStrategy


class HelpStrategy(IStrategy):

    def __init__(self, logger):
        self.logger = logger

    def execute(self):
        self.logger.write_message("This project contains three different strategies:\n"

                                  "* Save Database:\n"
                                  "\tThis strategy will save all the images inside the directory "
                                  "Assets/Dataset/Gesture_image_data in two pickles (test and train). To execute it "
                                  "you need the following arguments:\n "
                                  "\t\t--saveDatabase <string> <boolean> <boolean>\n"
                                  "\tThe string has to contain the name of the pickel to use, for example:\n"
                                  "\t\t\tsign_gesture_gray_150x150px\n"
                                  "\tThe first boolean will be true if the directory Gesture_image_data contains the "
                                  "images separated in test and train.\n"
                                  "\tThe second boolean will be true if the dataset has to be saved in gray colors.\n\n"

                                  "* Train Neural Network:\n"
                                  "\tThis strategy will train an specific neural network based on one of the models "
                                  "stored in the Dataset/Pickels. To execute it you need the following arguments:\n "
                                  "\t\t--trainNeuralNetwork <string> <string>\n"
                                  "\tThe first string has to contain the name of the directory storing the pickels. "
                                  "For example, if the pickel you want to use is in the path:\n "
                                  "\t\t\tAssets/Dataset/Pickels/sign_gesture_gray_150x150px"
                                  "/sign_gesture_gray_150x150px_train.pkl\n "
                                  "\tYou will have to enter:\n"
                                  "\t\t\tsign_gesture_gray_150x150px\n"
                                  "\tThe second string specifies the type of Neural Network to train, the possibilities"
                                  " are:\n"
                                  "\t\t* nn: Basic Neural Network.\n "
                                  "\t\t* cnn: Convolutional Neural Network.\n\n"

                                  "* Accuracy Neural Network:\n"
                                  "\tThis strategy will show the accuracy of the Neural Network selected. In order to "
                                  "be able to do it, it will need to execute the Train Neural Network before, so a "
                                  "model is stored in the Assets/NeuralNetworkModel directory. To execute it you need "
                                  "the following arguments:\n "
                                  "\t\t--accuracyNeuralNetwork <string> <string>\n"
                                  "\tThe first string has to contain the name of the directory storing the pickels. "
                                  "For example, if the pickel you want to use is in the path:\n "
                                  "\t\t\tAssets/Dataset/Pickels/sign_gesture_gray_150x150px"
                                  "/sign_gesture_gray_150x150px_train.pkl\n "
                                  "\tYou will have to enter:\n"
                                  "\t\t\tsign_gesture_gray_150x150px\n"
                                  "\tThe second string specifies the type of Neural Network to use, the possibilities "
                                  "are:\n"
                                  "\t\t* nn: Basic Neural Network.\n "
                                  "\t\t* cnn: Convolutional Neural Network.\n\n"

                                  "* Help:\n"
                                  "\tThis strategy will show all the needed arguments information in order to run "
                                  "this project. It needs the argument:\n "
                                  "\t\t--help")

