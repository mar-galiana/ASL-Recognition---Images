from Logger.iLogger import ILogger


class Logger(ILogger):

    def write_message(self, message):
        print(message)

    def write_info(self, message):
        print("[INFO]: " + message)

    def write_error(self, message):
        print("[ERROR]: " + message)
