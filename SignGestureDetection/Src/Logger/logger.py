from Logger.iLogger import ILogger


class Logger(ILogger):

    @staticmethod
    def write_message(message):
        print(message)

    @staticmethod
    def write_info(message):
        print("[INFO]: " + message)

    @staticmethod
    def write_error(message):
        print("[ERROR]: " + message)

    @staticmethod
    def write_action_required(message):
        print("[ACTION REQUIRED]: " + message)
