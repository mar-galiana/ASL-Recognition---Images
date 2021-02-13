class ParametersMissing(Exception):
    def __init__(self, number_parameters=1):
        super().__init__(f"Parameters missing, {number_parameters} must be entered")

