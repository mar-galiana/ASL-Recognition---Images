from enum import Enum
from Exception.inputException import InputException
from ExectuionFactory.saveDatabaseStrategy import SaveDatabaseStrategy
from ExectuionFactory.executeAlgorithmStrategy import ExecuteAlgorithmStrategy


class ExecutionFactory:

    def __init__(self, strategy, arguments):
        self.execution_strategy = strategy
        self.arguments = arguments

        self.strategy_switcher = {
            Strategies.SAVE_DATABASE.value: lambda: self.save_database(),
            Strategies.EXECUTE_ALGORITHM.value: lambda: self.execute_algorithm()
        }

    def get_execution_strategy(self):
        if self.execution_strategy in self.strategy_switcher:
            strategy = self.strategy_switcher.get(self.execution_strategy)

        else:
            raise InputException(self.execution_strategy + " is not a valid strategy")

        return strategy

    @staticmethod
    def save_database():
        return SaveDatabaseStrategy()

    def execute_algorithm(self):

        if len(self.arguments) > 0:
            strategy = ExecuteAlgorithmStrategy(self.arguments)
        else:
            raise InputException("this strategy requires arguments to be executed")

        return strategy


class Strategies(Enum):
    SAVE_DATABASE = "--saveDatabase"
    EXECUTE_ALGORITHM = "--executeAlgorithm"
