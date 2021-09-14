from abc import ABC, abstractmethod


class INeuralNetwork(ABC):

    @abstractmethod
    def train_neural_network(self):
        pass
