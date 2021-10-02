from abc import ABC, abstractmethod


class INeuralNetwork(ABC):
    """
    Neural network interface
    """
    @abstractmethod
    def train_neural_network(self):
        pass
