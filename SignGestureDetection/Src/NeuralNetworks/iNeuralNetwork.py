from abc import ABC, abstractmethod


class INeuralNetwork(ABC):

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def resize_data(self, environment, shape):
        pass

