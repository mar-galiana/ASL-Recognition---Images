class IncorrectVariableType(Exception):
    """Exception raised for errors in the vatiable type.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message="incorrect variable type"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in variable type, " + message + "."
        super().__init__(self.message)


class IncorrectNumberOfParameters(Exception):
    """Exception raised for errors in the number of parameters.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message="incorrect number of parameters"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in variable type, " + message + "."
        super().__init__(self.message)
