from enum import Enum
from saveDatabaseStrategy import SaveDatabaseStrategy


class ExecutionFactory:

    def __init__(self, strategy):
        self.execution_strategy = strategy

        self.switcher = {
            Strategies.SAVE_DATABASE.value: lambda: self.save_database(),
        }

    def get_execution_strategy(self):

        strategy = self.switcher.get(self.execution_strategy, lambda: print("Error, Invalid Strategy"))
        return strategy

    def save_database(self):
        return SaveDatabaseStrategy()


class Strategies(Enum):
    SAVE_DATABASE = "--saveDatabase"
