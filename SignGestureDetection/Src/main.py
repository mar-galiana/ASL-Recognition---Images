import sys
from Src.Logger.logger import Logger
from Src.Exception.inputException import InputException
from Src.StrategyFactory.strategyFactoy import ExecutionFactory


if __name__ == '__main__':

    logger = Logger()

    try:
        if len(sys.argv) <= 1:
            raise InputException("A strategy needs to be defined in order to execute it")

        strategy_factory = ExecutionFactory(logger, sys.argv[1], sys.argv[2:])
        strategy = strategy_factory.get_execution_strategy()
        strategy.execute()

    except Exception as e:
        logger.write_error(e.message)


