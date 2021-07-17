from Src.StrategyFactory.iStrategy import IStrategy
from Src.Exception.inputOutputException import InputException


class SaveDatabaseStrategy(IStrategy):

    def __init__(self, logger, model, arguments):
        self.logger = logger
        self.model = model

        if not self.__is_boolean_argument_correct(arguments[1]) or not self.__is_boolean_argument_correct(arguments[2]):
            raise InputException("The argument of this execution needs to be true or false")

        self.pickel_name = arguments[0]
        self.environments_separated = (arguments[1] == "true")
        self.as_gray = (arguments[2] == "true")

    def execute(self):
        self.model.create_pickle(self.pickel_name, self.environments_separated, self.as_gray)
        self.logger.write_info("Test and Train pickles have been created")

    @staticmethod
    def __is_boolean_argument_correct(value):
        return value.lower() == "true" or value.lower() == "false"
