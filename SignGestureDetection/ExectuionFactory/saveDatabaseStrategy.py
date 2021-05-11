from Model.environment import Environment
from ExectuionFactory.iStrategy import IStrategy
from Exception.inputException import InputException


class SaveDatabaseStrategy(IStrategy):

    def __init__(self, model, arguments):
        self.model = model

        if arguments[0] != "true" and arguments[0] != "false":
            raise InputException("The argument of this execution needs to be true or false")

        self.environments_separated = (arguments[0] == "true")

    def execute(self):
        self.model.create_pickle(self.environments_separated)
