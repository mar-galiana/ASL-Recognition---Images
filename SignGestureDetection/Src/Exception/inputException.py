class InputException(Exception):
    """Exception raised for errors in the input arguments.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="incorrect input arguments"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in the input arguments, " + message + "."
        super().__init__(self.message)
