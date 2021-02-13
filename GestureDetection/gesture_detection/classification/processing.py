import skimage
from skimage.transform import resize
from skimage.transform import rescale
from gesture_detection.exception.error import ClassificationNotPrepared
from gesture_detection.classification.iclassification import IClassification


class Processing(IClassification):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def prepare(self, width, height):
        self.width = width
        self.height = height

    def perform(self, image):

        # if image is None:
        #     raise ClassificationNotPrepared()

        image = resize(image, (self.width, self.height))

        return image
