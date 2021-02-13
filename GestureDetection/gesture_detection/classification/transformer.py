import numpy as np
from sklearn.preprocessing import StandardScaler
from gesture_detection.classification.iclassification import IClassification
from gesture_detection.classification.rgb2GrayTransformer import RGB2GrayTransformer, HogTransformer


class Transformer(IClassification):
    PIXELS_PER_CELL = (14, 14)
    CELLS_PER_BLOCK = (2, 2)
    ORIENTATION = 9
    BLOCK_NORM = 'L2-Hys'

    def __init__(self):
        self.gray = self.hog = self.scaler = self.x_prepared = None

    def perform(self, data):
        # create an instance of each transformer
        x = np.array(data.get_data()['data'])

        self.gray = RGB2GrayTransformer()
        self.hog = HogTransformer(pixels_per_cell=self.PIXELS_PER_CELL, cells_per_block=self.CELLS_PER_BLOCK,
                                  orientations=self.ORIENTATION, block_norm=self.BLOCK_NORM)
        self.scaler = StandardScaler()

        # call fit_transform on each transform converting x step by step
        x_gray = self.gray.fit_transform(x)
        x_hog = self.hog.fit_transform(x_gray)
        self.x_prepared = self.scaler.fit_transform(x_hog)
