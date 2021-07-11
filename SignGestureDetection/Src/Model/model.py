import os
import joblib
import numpy as np
from skimage import io
from skimage.transform import resize
from Src.Model.enumerations import Environment, Image
from sklearn.model_selection import train_test_split
from Src.Exception.modelException import EnvironmentException
from Src.Exception.inputOutputException import PathDoesNotExistException


class Model:
    BASE_PATH = f"{os.getcwd()}/../Assets/Dataset/"
    BASE_NAME = "sign_gesture"
    DATASET_SRC = BASE_PATH + "Gesture_image_data/"
    PICKELS_SRC = BASE_PATH + "Pickels/"

    def __init__(self, width=150, height=None):
        self.__train_data = None
        self.__test_data = None
        self.width = width
        self.height = (height, width)[height is None]
        self.base_pickle_src = f"{self.PICKELS_SRC}{self.BASE_NAME}_%s_{self.width}x{self.height}px.pkl"

    def create_pickle(self, environments_separated):
        data = {
            'description': f"resized ({int(self.width)}x{int(self.height)}) sign images in rgb"
        }

        if environments_separated:
            data[Environment.TEST] = self.__read_images(self.DATASET_SRC + "test/")
            data[Environment.TRAIN] = self.__read_images(self.DATASET_SRC + "train/")
        else:
            images_data = self.__read_images(self.DATASET_SRC)
            data[Environment.TEST], data[Environment.TRAIN] = self.__split_data_into_test_and_train(images_data)

        self.__write_data_into_pickle(data[Environment.TRAIN][Image.DATA.value],
                                      data[Environment.TRAIN][Image.LABEL.value],
                                      Environment.TRAIN.value)
        self.__write_data_into_pickle(data[Environment.TEST][Image.DATA.value],
                                      data[Environment.TEST][Image.LABEL.value],
                                      Environment.TEST.value)

    @staticmethod
    def __split_data_into_test_and_train(data):
        x = np.array(data[Image.DATA.value])
        y = np.array(data[Image.LABEL.value])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

        test_data = {Image.DATA.value: x_test, Image.LABEL.value: y_test}
        train_data = {Image.DATA.value: x_train, Image.LABEL.value: y_train}

        return test_data, train_data

    def __read_images(self, path):
        images_data = {
            Image.LABEL.value: [],
            Image.DATA.value: []
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
                images_data[Image.LABEL.value].append(subdir)
                images_data[Image.DATA.value].append(image)

        return images_data

    def __write_data_into_pickle(self, x, y, environment):
        environment_data = {
            'description': f"resized ({int(self.width)}x{int(self.height)}) {environment}ing sign images in rgb ",
            Image.LABEL.value: y,
            Image.DATA.value: x
        }

        joblib.dump(environment_data, self.base_pickle_src % environment)

        if environment == Environment.TRAIN.value:
            self.__train_data = environment_data

    def __read_data(self, environment):

        pickle_src = self.base_pickle_src % environment.value

        if os.path.exists(pickle_src):
            data = joblib.load(pickle_src)
        else:
            raise PathDoesNotExistException("The pickle needs to exists before using it")

        return data

    def __get_data(self, environment):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if environment == Environment.TRAIN:
            if self.__train_data is None:
                self.__train_data = self.__read_data(environment)
            data = self.__train_data

        else:
            if self.__test_data is None:
                self.__test_data = self.__read_data(environment)
            data = self.__test_data

        return data

    def get_x(self, environment):
        data = self.__get_data(environment)
        return np.array(data[Image.DATA.value])

    def get_y(self, environment):
        data = self.__get_data(environment)
        return np.array(data[Image.LABEL.value])

    def set_x(self, environment, data):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if environment == Environment.TRAIN:
            self.__train_data[Image.DATA.value] = data
        else:
            self.__test_data[Image.DATA.value] = data

    def set_y(self, environment, label):
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        if environment == Environment.TRAIN:
            self.__train_data[Image.LABEL.value] = label
        else:
            self.__test_data[Image.LABEL.value] = label
