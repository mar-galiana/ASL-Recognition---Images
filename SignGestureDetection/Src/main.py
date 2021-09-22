import sys
import time
from Logger.logger import Logger
from Exception.inputOutputException import InputException
from StrategyFactory.strategyFactoy import ExecutionFactory


if __name__ == '__main__':

    # try:
    # Input control
    if len(sys.argv) < 2:
        raise InputException("A strategy needs to be defined in order to execute it")

    start_time = time.time()

    logger = Logger()
    strategy_factory = ExecutionFactory(logger, sys.argv[1], sys.argv[2:])

    # Execute the strtagey selected
    strategy = strategy_factory.get_execution_strategy()
    strategy.execute()

    logger.write_info("Execution finished")

    duration = time.gmtime(time.time() - start_time)
    logger.write_info("The duration of the execution has been " + time.strftime("%H:%M:%S", duration) + " [hh:mm:ss]")

    # except Exception as e:
   #     logger.write_error(str(e))
