from abc import ABC, abstractmethod


class INeuralNetwork(ABC):

    @abstractmethod
    def train_neural_network(self):
        pass

    @abstractmethod
    def resize_data(self, environment, shape):
        pass

