from abc import ABC, abstractmethod


class ILogger(ABC):

    @abstractmethod
    def write_info(self, message):
        pass

    @abstractmethod
    def write_message(self, message):
        pass

    @abstractmethod
    def write_error(self, message):
        pass
