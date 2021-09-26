import os
import gzip
import numpy as np
from skimage import io
import _pickle as cPickle
from skimage.transform import resize
from Constraints.path import IMAGES_PATH, PICKLES_PATH
from sklearn.model_selection import train_test_split
from Exception.modelException import DatasetException
from Model.modelEnum import Environment, Image, Dataset
from Exception.inputOutputException import PathDoesNotExistException


class OutputModel:
    """
    A class used to store the samples in the pickles' files.

    Attributes
    ----------
    width : number
        Width used to resize the images
    height : number
        Height used to resize the images
    
    Methods
    -------
    create_pickle(pickle_name, dataset, environments_separated)
        Write images from dataset into pickle
    load_image(src, as_gray=True)
        Read image stored in the source path
    """
    def __init__(self, width=150, height=None):
        """
        Parameters
        ----------
        width : number
            Width used to resize the images
        height : number, optional
            Height used to resize the images (default is None)
        """
        self.width = width
        self.height = (height, width)[height is None]

    def create_pickle(self, pickle_name, dataset, environments_separated):
        """Write images from dataset into pickle.

        Parameters
        ----------
        pickle_name : string
            Name of the file to create
        dataset : Dataset
            Name of the dataset containing the images to store in the pickle
        environments_separated : boolean
            Boolean indicating if the database contains the samples separated in test
            and train folders.
        """
        base_pickle_src = f"{PICKLES_PATH}{pickle_name}/"
        data = self.__get_data(dataset, environments_separated)

        if not os.path.isdir(base_pickle_src):
            os.mkdir(base_pickle_src)

        base_pickle_src = f"{base_pickle_src}{pickle_name}_%s.pkl"

        self.__write_data_into_pickle(data[Environment.TEST][Image.DATA.value],
                                      data[Environment.TEST][Image.LABEL.value],
                                      Environment.TEST.value,
                                      base_pickle_src)
        self.__write_data_into_pickle(data[Environment.TRAIN][Image.DATA.value],
                                      data[Environment.TRAIN][Image.LABEL.value],
                                      Environment.TRAIN.value,
                                      base_pickle_src)
    
    def load_image(self, src, as_gray=True):
        """Read image stored in the source path.

        Parameters
        ----------
        src : string
            Image's path
        as_gray : boolean
            Boolean indicating if the images will be tranform to a gray color scale
    
        Returns
        ----------
        array
            Array of the image's pixels 
        """
        if not os.path.exists(src):
            raise PathDoesNotExistException("Image " + src + " does not exist.")
        
        image = io.imread(src, as_gray=as_gray)
        image = resize(image, (self.width, self.height))
        return image

    def __get_data(self, dataset, environments_separated):

        if not isinstance(dataset, Dataset):
            raise DatasetException("Dataset selected is not a valid one")

        data = {
            Image.DESCRIPTION.value: f"resized ({int(self.width)}x{int(self.height)}) sign images from {dataset.value} "
                                     f"dataset. "
        }

        image_path = f"{IMAGES_PATH}{dataset.value}/"
        if environments_separated:
            data[Environment.TEST] = self.__read_images(image_path + "test/")
            data[Environment.TRAIN] = self.__read_images(image_path + "train/")
        else:
            images_data = self.__read_images(image_path)
            data[Environment.TEST], data[Environment.TRAIN] = self.__split_data_into_test_and_train(images_data)

        return data

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

        # read all images in path, resize and write to DESTINATION_PATH
        for subdir in os.listdir(path):
            current_path = os.path.join(path, subdir)

            if os.path.isfile(current_path):
                continue

            for file in os.listdir(current_path):
                if file.startswith("."):
                    continue

                src = os.path.join(current_path, file)
                image = self.load_image(src)
                images_data[Image.LABEL.value].append(subdir)
                images_data[Image.DATA.value].append(image)

        images_data[Image.LABEL.value] = np.array(images_data[Image.LABEL.value])
        images_data[Image.DATA.value] = np.array(images_data[Image.DATA.value])
        return images_data

    def __write_data_into_pickle(self, x, y, environment, base_pickle_src):
        environment_data = {
            Image.DESCRIPTION.value: f"resized ({int(self.width)}x{int(self.height)}) {environment}ing sign images.",
            Image.LABEL.value: y,
            Image.DATA.value: x
        }

        with gzip.open(base_pickle_src % environment, 'wb') as f:
            cPickle.dump(environment_data, f, -1)
