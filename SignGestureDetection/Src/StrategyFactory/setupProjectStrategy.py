from Constraints.path import *
from StrategyFactory.iStrategy import IStrategy


class SetupProjectStructure(IStrategy):

    def __init__(self, logger, storage_controller):
        self.logger = logger
        self.storage_controller = storage_controller

    def execute(self):
        self.storage_controller.create_directory(ASSETS_PATH)

        # Dataset directories and files
        self.storage_controller.create_directory(DATASET_PATH)
        self.storage_controller.create_directory(IMAGES_PATH)
        self.storage_controller.create_directory(PICKELS_PATH)
        self.storage_controller.create_json_file(SIGNS_FILE, {})

        # Model Structures directories and files
        self.storage_controller.create_directory(MODEL_STRUCTURES_PATH)
        self.storage_controller.create_directory(BINARY_NEURAL_NETWORK_MODEL_PATH)
        self.storage_controller.create_directory(TMP_BINARY_NEURAL_NETWORK_MODEL_PATH)
        self.storage_controller.create_directory(CATEGORICAL_NEURAL_NETWORK_MODEL_PATH)
        self.storage_controller.create_directory(DECISION_TREE_PATH)
        self.storage_controller.create_directory(DECISION_TREE_MODEL_PATH)
        self.storage_controller.create_directory(DECISION_TREE_PLOT_PATH)

        self.logger.write_action_required("The repository specified in the readme file must be cloned in the path: " +
                                          IMAGES_PATH)
        self.logger.write_action_required("Execute the command 'pip install -r requirements.txt' ")
        self.logger.write_info("Strategy executed successfully")
