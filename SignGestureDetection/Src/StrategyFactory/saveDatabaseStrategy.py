from Src.StrategyFactory.iStrategy import IStrategy
from Src.Exception.inputOutputException import InputException


class SaveDatabaseStrategy(IStrategy):

    def __init__(self, logger, model, arguments):
        self.logger = logger
        self.model = model

        if arguments[0] != "true" and arguments[0] != "false":
            raise InputException("The argument of this execution needs to be true or false")

        self.environments_separated = (arguments[0] == "true")

    def execute(self):
        self.model.create_pickle(self.environments_separated)
        self.logger.write_info("Test and Train pickles have been created")
