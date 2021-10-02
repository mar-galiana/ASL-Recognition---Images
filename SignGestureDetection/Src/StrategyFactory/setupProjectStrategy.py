from Constraints.path import *
from StrategyFactory.iStrategy import IStrategy


class SetupProjectStructure(IStrategy):
    """
    A class to setup the project by creating the directories and the files needed in all the strategies

    Attributes
    ----------
    logger : Logger
        A class used to show the execution information
    storage_controller : StorageController
        A class used to remove and create the directories and files used in the execution

    Methods
    -------
    execute()
        Create the directories and the files needed to execute the strategies
    """

    def __init__(self, logger, storage_controller):
        self.logger = logger
        self.storage_controller = storage_controller

    def execute(self):
        """Create the directories and the files needed to execute the strategies
        """

        self.__create_assets_directory()
        self.__create_datasets_directories()
        self.__create_structures_directories()

        self.logger.write_action_required("The repository specified in the readme file must be cloned in the path: " +
                                          IMAGES_PATH)
        self.logger.write_action_required("Execute the command 'pip install -r requirements.txt' ")
        self.logger.write_info("Strategy executed successfully")
    
    def __create_assets_directory(self):
        self.storage_controller.create_directory(ASSETS_PATH)
        self.logger.write_info(ASSETS_PATH + " directory created")
    
    def __create_datasets_directories(self):
        self.storage_controller.create_directory(DATASET_PATH)
        self.logger.write_info(DATASET_PATH + " directory created")
        
        self.storage_controller.create_directory(IMAGES_PATH)
        self.logger.write_info(IMAGES_PATH + " directory created")
        
        self.storage_controller.create_directory(SIGNS_IMAGES)
        self.logger.write_info(SIGNS_IMAGES + " directory created")
        
        self.storage_controller.create_directory(PICKLES_PATH)
        self.logger.write_info(PICKLES_PATH + " directory created")
        
        self.storage_controller.create_json_file(SIGNS_FILE, {})
        self.logger.write_info(SIGNS_FILE + " file created")

    def __create_structures_directories(self):
        self.storage_controller.create_directory(MODEL_STRUCTURES_PATH)
        self.logger.write_info(MODEL_STRUCTURES_PATH + " directory created")
        
        self.storage_controller.create_directory(BINARY_NEURAL_NETWORK_MODEL_PATH)
        self.logger.write_info(BINARY_NEURAL_NETWORK_MODEL_PATH + " directory created")
        
        self.storage_controller.create_directory(TMP_BINARY_NEURAL_NETWORK_MODEL_PATH)
        self.logger.write_info(TMP_BINARY_NEURAL_NETWORK_MODEL_PATH + " directory created")
        
        self.storage_controller.create_directory(CATEGORICAL_NEURAL_NETWORK_MODEL_PATH)
        self.logger.write_info(CATEGORICAL_NEURAL_NETWORK_MODEL_PATH + " directory created")
       
        self.storage_controller.create_directory(DECISION_TREE_PATH)
        self.logger.write_info(DECISION_TREE_PATH + " directory created")

        self.storage_controller.create_directory(DECISION_TREE_MODEL_PATH)
        self.logger.write_info(DECISION_TREE_MODEL_PATH + " directory created")

        self.storage_controller.create_directory(DECISION_TREE_PLOT_PATH)
        self.logger.write_info(DECISION_TREE_PLOT_PATH + " directory created")