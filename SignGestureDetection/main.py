import sys
from ExectuionFactory.strategyFactoy import ExecutionFactory


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Error, you need to introduce the model to execute")

    else:
        strategy_factory = ExecutionFactory(sys.argv[1], sys.argv[2:])
        strategy = strategy_factory.get_execution_strategy()
        strategy()

