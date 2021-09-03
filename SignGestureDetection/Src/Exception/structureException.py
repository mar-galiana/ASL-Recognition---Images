class StructureException(Exception):
    """Exception raised for errors in the model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="incorrect structure"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in the structure, " + message + "."
        super().__init__(self.message)


class LabelsRequirementException(Exception):
    """Exception raised for errors in the model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="incorrect label requirement"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in the structure, " + message + "."
        super().__init__(self.message)


class StructureFileElementDoesNotExists(Exception):
    """Exception raised for errors in the model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="element does not exist"):
        if len(message) > 0:
            message = message[0].lower() + message[1:]

        self.message = "Error in structure file, " + message + "."
        super().__init__(self.message)
