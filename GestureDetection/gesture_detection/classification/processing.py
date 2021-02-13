import skimage
from skimage.transform import resize
from skimage.transform import rescale
from gesture_detection.exception.error import ClassificationNotPrepared
from gesture_detection.classification.iclassification import IClassification


class Processing(IClassification):

    def __init__(self):
        self.scale = 1 / 3
        self.image = self.width = self.height = None

    def prepare(self, image, width, height):
        self.image = image
        self.width = width
        self.height = height

    def perform(self):
        """
        image_hog, image_hog_img = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                       visualize=True)

        print('number of pixels: ', image.shape[0] * image.shape[1])
        print('number of hog features: ', image_hog.shape[0])
        """

        if self.image is None:
            raise ClassificationNotPrepared()

        image = resize(self.image, (self.width, self.height))
        self.image = None

        return image
