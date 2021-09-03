class EnvironmentException(Exception):
    """Exception raised for errors in the model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="incorrect environment"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in the model, " + message + "."
        super().__init__(self.message)


class DatasetException(Exception):
    """Exception raised for errors in the model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="incorrect dataset"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in the model, " + message + "."
        super().__init__(self.message)


class SignIsNotInJsonFileException(Exception):
    """Exception raised for errors in the model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="sign does not exist"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in the model, " + message + "."
        super().__init__(self.message)


class DifferentPickelsException(Exception):
    """Exception raised for errors in the model.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message="models selected use different pickels"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in the model, " + message + "."
        super().__init__(self.message)


class SignsFileHasNotBeenReadException(Exception):
    """Exception raised for errors in the model.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message="sign json file hasn't been read"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in the model, " + message + "."
        super().__init__(self.message)
