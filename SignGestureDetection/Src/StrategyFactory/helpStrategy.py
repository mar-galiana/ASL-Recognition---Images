from Src.StrategyFactory.iStrategy import IStrategy


class HelpStrategy(IStrategy):

    def __init__(self, logger):
        self.logger = logger

    def execute(self):
        self.logger.write_message("This project contains three different strategies:\n"

                                  "* Save Database:\n"
                                  "\tThis strategy will save all the images inside the directory "
                                  "Assets/Dataset/Gesture_image_data in two pickles (test and train). To execute it "
                                  "you need the following arguments:\n "
                                  "\t\t--saveDatabase <boolean>\n"
                                  "\tThe boolean will be true if the directory Gesture_image_data contains the images "
                                  "separated in test and train\n\n"

                                  "* Train Neural Network:\n"
                                  "\tThis strategy will train an specific neural network based on one of the models "
                                  "stored in the Dataset/Pickels. To execute it you need the following arguments:\n "
                                  "\t\t--trainNeuralNetwork <string>\n"
                                  "\tThe string specifies the type of Neural Network to train, the possibilities are:\n"
                                  "\t\t* nn: Basic Neural Network.\n "
                                  "\t\t* cnn: Convolutional Neural Network.\n\n"

                                  "* Accuracy Neural Network:\n"
                                  "\tThis strategy will show the accuracy of the Neural Network selected. In order to "
                                  "be able to do it, it will need to execute the Train Neural Network before, so a "
                                  "model is stored in the Assets/NeuralNetworkModel directory. To execute it you need "
                                  "the following arguments:\n "
                                  "\t\t--accuracyNeuralNetwork <string>\n"
                                  "\tThe string specifies the type of Neural Network to use, the possibilities are:\n"
                                  "\t\t* nn: Basic Neural Network.\n "
                                  "\t\t* cnn: Convolutional Neural Network.\n\n"

                                  "* Help:\n"
                                  "\tThis strategy will show all the needed arguments information in order to run "
                                  "this project. It needs the argument:\n "
                                  "\t\t--help")

