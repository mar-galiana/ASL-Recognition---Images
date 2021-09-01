import os

# ASSETS_PATH = os.getcwd() + "/../Assets/"
ASSETS_PATH = os.getcwd() + "/Assets/"

# Dataset
DATASET_PATH = ASSETS_PATH + "Dataset/"
PICKELS_PATH = DATASET_PATH + "Pickels/"
IMAGES_PATH = DATASET_PATH + "Images/"
SIGNS_FILE = IMAGES_PATH + "signs.json"

# Model Structures
MODEL_STRUCTURES_PATH = ASSETS_PATH + "ModelStructures/"
MODEL_PICKELS_FILE = MODEL_STRUCTURES_PATH + "model_pickels.json"

# Model Structures --> Categorical Neural Network
CATEGORICAL_NEURAL_NETWORK_MODEL_PATH = MODEL_STRUCTURES_PATH + "CategoricalNeuralNetworkModel/"

# Model Structures --> Binary Neural Network
BINARY_NEURAL_NETWORK_MODEL_PATH = MODEL_STRUCTURES_PATH + "BinaryNeuralNetworkModel/"
TMP_BINARY_NEURAL_NETWORK_MODEL_PATH = BINARY_NEURAL_NETWORK_MODEL_PATH + "Tmp/"

# Model Structures --> Decision Tree
DECISION_TREE_PATH = MODEL_STRUCTURES_PATH + "DecisionTreeModel/"
DECISION_TREE_MODEL_PATH = DECISION_TREE_PATH + "Model/"
DECISION_TREE_PLOT_PATH = DECISION_TREE_PATH + "TreePlot/"
