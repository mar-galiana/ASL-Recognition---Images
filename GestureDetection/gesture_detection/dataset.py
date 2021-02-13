import os
import joblib
import skimage
from skimage.io import imread
from classification.processing import Processing


class Dataset:

    BASE_PATH = f"{os.getcwd()}/assets/"
    BASE_NAME = "sign_gesture"

    def __init__(self, src, environment="train", width=150, height=None):
        self.src = f"{os.getcwd()}/{src}"
        self.__data = None
        self.width = width
        self.environment = environment
        self.height = (height, width)[height is None]
        self.pickle_src = f"{self.BASE_PATH}{self.BASE_NAME}_{self.environment}_{self.width}x{self.height}px.pkl"

    def __write_pickle(self):
        self.__data = {
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
                image = imread(src, as_gray=True)
                # processing.prepare(image, self.width, self.height)
                image = processing.perform(image)
                self.__data['label'].append(subdir)
                self.__data['data'].append(image)

        joblib.dump(self.__data, self.pickle_src)

    def __read_pickle(self):
        self.__data = joblib.load(self.pickle_src)

    def get_data(self):

        if self.__data is None:

            if os.path.exists(self.pickle_src):
                self.__read_pickle()
            else:
                self.__write_pickle()

        return self.__data
