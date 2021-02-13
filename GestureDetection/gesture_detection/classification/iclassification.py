import abc
from abc import ABC, abstractmethod


class IClassification(ABC):

    @abstractmethod
    def perform(self, information):
        pass
