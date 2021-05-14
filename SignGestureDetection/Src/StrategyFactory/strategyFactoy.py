from enum import Enum
from Src.Model.model import Model
from Src.Exception.inputException import InputException
from Src.StrategyFactory.helpStrategy import HelpStrategy
from Src.StrategyFactory.saveDatabaseStrategy import SaveDatabaseStrategy
from Src.StrategyFactory.executeAlgorithmStrategy import ExecuteAlgorithmStrategy


class ExecutionFactory:

    def __init__(self, logger, strategy, arguments):
        self.logger = logger
        self.execution_strategy = strategy
        self.arguments = arguments
        self.model = Model()

        self.strategy_switcher = {
            Strategies.HELP.value: lambda: self.help(),
            Strategies.SAVE_DATABASE.value: lambda: self.save_database(),
            Strategies.EXECUTE_ALGORITHM.value: lambda: self.execute_algorithm()
        }

    def get_execution_strategy(self):
        if self.execution_strategy not in self.strategy_switcher:
            raise InputException(self.execution_strategy + " is not a valid strategy")

        strategy_method = self.strategy_switcher.get(self.execution_strategy)
        return strategy_method()

    def save_database(self):
        if len(self.arguments) != 1:
            raise InputException("This strategy requires arguments to be executed")

        return SaveDatabaseStrategy(self.logger, self.model, self.arguments)

    def execute_algorithm(self):

        if len(self.arguments) != 1:
            raise InputException("This strategy requires arguments to be executed")

        return ExecuteAlgorithmStrategy(self.logger, self.arguments)

    def help(self):
        return HelpStrategy(self.logger)


class Strategies(Enum):
    HELP = "--help"
    SAVE_DATABASE = "--saveDatabase"
    EXECUTE_ALGORITHM = "--executeAlgorithm"
