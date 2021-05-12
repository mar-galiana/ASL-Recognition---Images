import os
import joblib
import numpy as np
from skimage import io
from skimage.transform import resize
from Src.Model.environment import Environment
from sklearn.model_selection import train_test_split
from Src.Exception.modelException import EnvironmentException


class Model:

    BASE_PATH = f"{os.getcwd()}/Assets/"
    BASE_NAME = "sign_gesture"
    DATASET_SRC = BASE_PATH + "Dataset/Gesture_image_data/"

    def __init__(self, width=150, height=None):
        self.__data = None
        self.width = width
        self.height = (height, width)[height is None]
        self.base_pickle_src = f"{self.BASE_PATH}{self.BASE_NAME}_%s_{self.width}x{self.height}px.pkl"

    def create_pickle(self, environments_separated):
        data = {
            'description': f"resized ({int(self.width)}x{int(self.height)}) sign images in rgb"
        }

        if environments_separated:
            data['test'] = self.__read_images(self.DATASET_SRC + "test/")
            data['train'] = self.__read_images(self.DATASET_SRC + "train/")
        else:
            images_data = self.__read_images(self.DATASET_SRC)
            data['test'], data['train'] = self.__split_data_into_test_and_train(images_data)

        self.__write_data_into_pickle(data['train']['data'], data['train']['label'], Environment.TRAIN.value)
        self.__write_data_into_pickle(data['test']['data'], data['test']['label'], Environment.TEST.value)

    @staticmethod
    def __split_data_into_test_and_train(data):
        x = np.array(data['data'])
        y = np.array(data['label'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

        test_data = {'data': x_test, 'label': y_test}
        train_data = {'data': x_train, 'label': y_train}

        return test_data, train_data

    def __read_images(self, path):
        images_data = {
            'label': [],
            'data': []
        }

        # read all images in PATH, resize and write to DESTINATION_PATH
        for subdir in os.listdir(path):
            current_path = os.path.join(path, subdir)

            if os.path.isfile(current_path):
                continue

            for file in os.listdir(current_path):
                src = os.path.join(current_path, file)
                image = io.imread(src, as_gray=True)
                image = resize(image, (self.width, self.height))
                images_data['label'].append(subdir)
                images_data['data'].append(image)

        return images_data

    def __write_data_into_pickle(self, x, y, environment):
        environment_data = {
            'description': f"resized ({int(self.width)}x{int(self.height)}) {environment}ing sign images in rgb ",
            'label': y,
            'data': x
        }

        joblib.dump(environment_data, self.base_pickle_src % environment)

        if environment == Environment.TRAIN.value:
            self.__data = environment_data

    def __read_pickle(self, pickle_src):
        self.__data = joblib.load(pickle_src)

    def get_data(self, environment):

        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if self.__data is None:
            pickle_src = self.base_pickle_src % environment

            if os.path.exists(pickle_src):
                self.__read_pickle(pickle_src)
            else:
                raise EnvironmentException("The pickle needs to exists before using it")

        return self.__data

    def get_x(self, environment):

        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        return np.array(self.get_data(environment)['data'])

    def get_y(self, environment):

        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        return np.array(self.get_data(environment)['label'])
