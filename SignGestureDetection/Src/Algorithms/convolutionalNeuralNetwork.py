import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

from Src.Algorithms.iAlgorithm import IAlgorithm


class ConvolutionalNeuralNetwork(IAlgorithm):

    def __init__(self, logger, model):
        self.logger = logger
        self.model = model
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None

    def execute(self):
        self.analyze_data()

    def analyze_data(self):
        self.logger.write_info("CNN executed")
        (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
        print('Training data shape : ', train_X.shape, train_Y.shape)
        print('Testing data shape : ', test_X.shape, test_Y.shape)
        classes = np.unique(train_Y)
        n_classes = len(classes)
        print('Total number of outputs : ', n_classes)
        print('Output classes : ', classes)

        plt.figure(figsize=[5, 5])

        # Display the first image in training data
        plt.subplot(121)
        plt.imshow(train_X[0, :, :], cmap='gray')
        plt.title("Ground Truth : {}".format(train_Y[0]))

        # Display the first image in testing data
        plt.subplot(122)
        plt.imshow(test_X[0, :, :], cmap='gray')
        plt.title("Ground Truth : {}".format(test_Y[0]))
        plt.show()
