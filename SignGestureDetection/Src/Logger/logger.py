from Logger.iLogger import ILogger
from enum import Enum


class Logger(ILogger):
    """
    A class used to show the execution information

    Methods
    -------
    write_message(message)
        Print message
    write_info(message)
        Print informative message
    write_error(message)
        Print error message
    write_action_required(message)
        Print action required message
    """

    @staticmethod
    def write_message(message):
        """Print message.

        Parameters
        ----------
        message : string
            Message to print
        """
        print(message)

    @staticmethod
    def write_info(message):
        """Print informative message.

        Parameters
        ----------
        message : string
            Message to print
        """
        print("[INFO]: " + message)

    @staticmethod
    def write_error(message):
        """Print error message.

        Parameters
        ----------
        message : string
            Message to print
        """
        print(Color.RED + "[ERROR]: " + message + Color.END)

    @staticmethod
    def write_action_required(message):
        """Print action required message.

        Parameters
        ----------
        message : string
            Message to print
        """
        print("[ACTION REQUIRED]: " + message)


class Color(Enum):
    """
    This enumeration is used to change the color and the format of the text that will be printed to the console.
    """

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
