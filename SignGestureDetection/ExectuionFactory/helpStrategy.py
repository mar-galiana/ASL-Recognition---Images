from ExectuionFactory.iStrategy import IStrategy


class HelpStrategy(IStrategy):

    def __init__(self, logger):
        self.logger = logger

    def execute(self):
        self.logger.write_message("This project contains three different strategies:\n"
                                  "* Execute Algorithm:\n"
                                  "\n\n"

                                  "* Save Database:\n"
                                  "\tThis strategy will save all the images inside the directory "
                                  "Assets/Dataset/Gesture_image_data in two pickles (test and train). To execute it "
                                  "you need the following arguments:\n "
                                  "\t\t--saveDatabase <boolean>\n"
                                  "\tThe boolean will be true if the directory Gesture_image_data contains the images "
                                  "separated in test and train\n\n "

                                  "* Help:\n"
                                  "\tThis strategy will show all the needed arguments information in order to run "
                                  "this project. It needs the argument:\n "
                                  "\t\t--help")
