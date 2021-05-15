from Src.Algorithms.iAlgorithm import IAlgorithm


class ConvolutionalNeuralNetwork(IAlgorithm):

    def __init__(self, logger, model):
        self.logger = logger
        self.model = model

    def execute(self):
        self.logger.write_info("CNN executed")
