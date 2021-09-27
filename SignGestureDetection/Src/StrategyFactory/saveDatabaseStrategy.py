from Model.modelEnum import Dataset
from StrategyFactory.iStrategy import IStrategy
from Exception.inputOutputException import InputException


class SaveDatabaseStrategy(IStrategy):
    """
    A class to save the database into two pickles

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    pickle_name : string
        Name of the pickle to create
    dataset : Dataset
        Dataset selected to save in a Pickle
    environments_separated : boolean
        Boolean indicating if the dataset is separated into testing and tested

    Methods
    -------
    execute()
        Save the database selected into two pickles, one with the testing samples and the other 
        one for the training ones.
    """

    def __init__(self, logger, model, arguments):
        """
        Parameters
        ----------
        logger : Logger
            A class used to show the execution information.
        model : Model
            A class used to sync up all the functionalities that refer to the database
        arguments: array
            Array of arguments entered without the execution strategy
        """

        arguments[1] = arguments[1]
        self.__check_input_values(arguments)

        self.logger = logger
        self.model = model
        self.__show_arguments_entered(arguments)

        self.pickle_name = arguments[0]
        self.dataset = Dataset(arguments[1])
        self.environments_separated = (arguments[2] == "true")

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Pickle's name: " + arguments[0] + "\n" \
                         "\t* Dataset: " + arguments[1] + "\n" \
                         "\t* Environments separated: " + arguments[2]
        self.logger.write_info(info_arguments)

    def __check_input_values(self, arguments):
        if arguments[1] not in list(map(lambda c: c.value, Dataset)):
            raise InputException("Check help strategy to know the possible datasets to use")

        if not self.__is_boolean_argument_correct(arguments[2]):
            raise InputException("The third argument of this execution needs to be true or false")

    def execute(self):
        """Save the database selected into two pickles, one with the testing samples and the other 
        one for the training ones.
        """
        self.model.create_pickle(self.pickle_name, self.dataset, self.environments_separated)
        self.logger.write_info("Test and Train pickles have been created")

    @staticmethod
    def __is_boolean_argument_correct(value):
        return value.lower() == "true" or value.lower() == "false"
