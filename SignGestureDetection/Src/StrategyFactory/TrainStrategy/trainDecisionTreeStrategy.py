from StrategyFactory.iStrategy import IStrategy
from Structures.DecisionTree.decisionTree import DecisionTree


class TrainDecisionTreeStrategy(IStrategy):
    """
    A class to train a decision tree model

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    model : Model
        A class used to sync up all the functionalities that refer to the database
    dt_util : DecisionTreeUtil
        TODO
    pickles : array
        Array of pickles to use in the training
    decisionTree : DecisionTree
        TODO

    Methods
    -------
    execute()
        To train a decision tree model using the training samples of the pickle selected
    """
    def __init__(self, logger, model, dt_util, arguments):
        """
        logger : Logger
            A class used to show the execution information
        model : Model
            A class used to sync up all the functionalities that refer to the database
        dt_util : DecisionTreeUtil
            TODO
        accuracy_util : AccuracyUtil
            TODO
        arguments : array
            Array of arguments entered in the execution
        """
        self.logger = logger
        self.model = model
        self.dt_util = dt_util
        self.__show_arguments_entered(arguments)

        self.pickles = arguments
        self.decisionTree = DecisionTree(self.logger, self.model, self.dt_util)

    def __show_arguments_entered(self, arguments):
        info_arguments = "Arguments entered:\n" \
                         "\t* Pickles selected: " + ", ".join(arguments)
        self.logger.write_info(info_arguments)

    def execute(self):
        """To train a decision tree model using the training samples of the pickle selected
        """

        self.model.set_pickles_name(self.pickles)
        
        xgboost_model = self.decisionTree.train_model()
        self.logger.write_info("Finished training the decision tree")

        self.dt_util.save_model(xgboost_model)
        self.logger.write_info("Strategy executed successfully")
