class ClassificationNotPrepared(Exception):
    def __init__(self):
        super().__init__("A classification was used without initializing it first")

