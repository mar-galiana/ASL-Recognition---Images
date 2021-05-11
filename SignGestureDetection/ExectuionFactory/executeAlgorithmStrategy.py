from ExectuionFactory.iStrategy import IStrategy


class ExecuteAlgorithmStrategy(IStrategy):

    def __init__(self, arguments):
        self.arguments = arguments

    def execute(self):
        pass

