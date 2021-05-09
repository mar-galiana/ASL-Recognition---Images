import sys
from ExectuionFactory.executionFactoy import ExecutionFactory


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Error, you need to introduce the model to execute")

    else:
        execution_factory = ExecutionFactory(sys.argv[1])
        strategy = execution_factory.get_execution_strategy()
        strategy()

