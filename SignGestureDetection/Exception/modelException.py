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
