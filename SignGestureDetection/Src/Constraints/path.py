"""
Path of the folders and files used in the project 
"""

import os

ASSETS_PATH = os.getcwd() + "/../Assets/"
# ASSETS_PATH = os.getcwd() + "/Assets/"

# Dataset
DATASET_PATH = ASSETS_PATH + "Dataset/"
PICKLES_PATH = DATASET_PATH + "Pickles/"
IMAGES_PATH = DATASET_PATH + "Images/"
SIGNS_IMAGES = DATASET_PATH + "Signs/"
SIGNS_FILE = DATASET_PATH + "signs.json"

# Model Structures
MODEL_STRUCTURES_PATH = ASSETS_PATH + "ModelStructures/"
MODEL_PICKLES_FILE = MODEL_STRUCTURES_PATH + "model_pickles.json"

# Model Structures --> Categorical Neural Network
CATEGORICAL_NEURAL_NETWORK_MODEL_PATH = MODEL_STRUCTURES_PATH + "CategoricalNeuralNetworkModel/"

# Model Structures --> Binary Neural Network
BINARY_NEURAL_NETWORK_MODEL_PATH = MODEL_STRUCTURES_PATH + "BinaryNeuralNetworkModel/"
TMP_BINARY_NEURAL_NETWORK_MODEL_PATH = BINARY_NEURAL_NETWORK_MODEL_PATH + "Tmp/"

# Model Structures --> Decision Tree
DECISION_TREE_PATH = MODEL_STRUCTURES_PATH + "DecisionTreeModel/"
DECISION_TREE_MODEL_PATH = DECISION_TREE_PATH + "Model/"
DECISION_TREE_PLOT_PATH = DECISION_TREE_PATH + "TreePlot/"
