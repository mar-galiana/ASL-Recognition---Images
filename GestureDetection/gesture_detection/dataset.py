import os
import joblib
import numpy as np
from skimage import io
from enumeration import Environment
from classification.processing import Processing
from sklearn.model_selection import train_test_split


class Dataset:

    BASE_PATH = f"{os.getcwd()}/assets/"
    BASE_NAME = "sign_gesture"
    DATASET_SRC = "dataset/gesture_image_data/"

    def __init__(self, environment=Environment.TRAIN, width=150, height=None):
        self.__data = None
        self.width = width
        self.environment = environment.value
        self.height = (height, width)[height is None]
        self.src = f"{os.getcwd()}/{self.DATASET_SRC}"
        self.base_pickle_src = f"{self.BASE_PATH}{self.BASE_NAME}_%s_{self.width}x{self.height}px.pkl"
        self.pickle_src = self.base_pickle_src % self.environment

    def __write_pickle(self):
        data = {
            'description': f"resized ({int(self.width)}x{int(self.height)}) sign images in rgb",
            'label': [],
            'data': []
        }
        processing = Processing(self.width, self.height)

        # read all images in PATH, resize and write to DESTINATION_PATH
        for subdir in os.listdir(self.src):
            current_path = os.path.join(self.src, subdir)

            if os.path.isfile(current_path):
                continue

            for file in os.listdir(current_path):
                src = os.path.join(current_path, file)
                image = io.imread(src, as_gray=True)
                image = processing.perform(image)
                data['label'].append(subdir)
                data['data'].append(image)

        x = np.array(data['data'])
        y = np.array(data['label'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

        self.__write_data_pickle(x_train, y_train, Environment.TRAIN.value)
        self.__write_data_pickle(x_test, y_test, Environment.TEST.value)

    def __write_data_pickle(self, x, y, environment):
        environment_data = {
            'description': f"resized ({int(self.width)}x{int(self.height)}) {environment}ing sign images in rgb ",
            'label': y,
            'data': x
        }

        joblib.dump(environment_data, self.base_pickle_src % environment)

        if environment == Environment.TRAIN.value:
            self.__data = environment_data

    def __read_pickle(self):
        self.__data = joblib.load(self.pickle_src)

    def get_data(self):

        if self.__data is None:

            if os.path.exists(self.pickle_src):
                self.__read_pickle()
            else:
                self.__write_pickle()

        return self.__data
