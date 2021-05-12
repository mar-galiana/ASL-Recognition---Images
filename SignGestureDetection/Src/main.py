import sys
from Src.Logger.Logger import Logger
from Src.Exception.inputException import InputException
from Src.StrategyFactory.strategyFactoy import ExecutionFactory


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        raise InputException("a strategy needs to be defined in order to execute it")

    else:

        logger = Logger()
        strategy_factory = ExecutionFactory(logger, sys.argv[1], sys.argv[2:])
        strategy = strategy_factory.get_execution_strategy()
        strategy.execute()

