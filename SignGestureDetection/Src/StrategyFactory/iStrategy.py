from abc import ABC, abstractmethod


class IStrategy(ABC):
    """
    Strategy execution interface
    """
    
    @abstractmethod
    def execute(self):
        pass

