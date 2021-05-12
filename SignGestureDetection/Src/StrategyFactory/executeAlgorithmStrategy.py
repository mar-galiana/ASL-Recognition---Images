from Src.StrategyFactory.iStrategy import IStrategy


class ExecuteAlgorithmStrategy(IStrategy):

    def __init__(self, logger, arguments):
        self.logger = logger
        self.arguments = arguments

    def execute(self):
        self.logger.write_info("Algorithm executed successfully")

