from Logger.iLogger import ILogger


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
        print("[ERROR]: " + message)

    @staticmethod
    def write_action_required(message):
        """Print action required message.

        Parameters
        ----------
        message : string
            Message to print
        """
        print("[ACTION REQUIRED]: " + message)
