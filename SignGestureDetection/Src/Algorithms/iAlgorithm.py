from abc import ABC, abstractmethod


class IAlgorithm(ABC):

    @abstractmethod
    def execute(self):
        pass

