import sys
from Exception.inputException import InputException
from ExectuionFactory.strategyFactoy import ExecutionFactory


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        raise InputException("a strategy needs to be defined in order to execute it")

    else:
        strategy_factory = ExecutionFactory(sys.argv[1], sys.argv[2:])
        strategy = strategy_factory.get_execution_strategy()
        strategy.execute()

