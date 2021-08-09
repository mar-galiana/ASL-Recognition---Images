import numpy as np
from unittest import TestCase
from Src.Model.model import Model
from Src.Logger.logger import Logger
from unittest.mock import Mock
from Src.Model.enumerations import Environment
from Src.Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil
from Src.Structures.NeuralNetworks.convolutionalNeuralNetwork import ConvolutionalNeuralNetwork


class TestConvolutionalNeuralNetwork(TestCase):

    def setUp(self):
        self.model = Mock(Model)
        self.model.get_x = Mock(return_value=np.ones((38850, 150, 150), dtype=int))
        self.logger = Mock(Logger)
        self.nn_util = Mock(NeuralNetworkUtil)
        self.nn_util.convert_to_one_hot_data = Mock(return_value=0)
        self.con_neural_network = ConvolutionalNeuralNetwork(self.logger, self.model, self.nn_util)

    def tearDown(self):
        self.model = None
        self.logger = None
        self.nn_util = None
        self.neural_network = None

    def test_WhenResizeDataIsCalled_WhileClassWorksAsExpected_ThenGetXIsCalledTwice(self):
        self.con_neural_network.resize_data(Environment.TRAIN)
        self.assertEqual(self.model.get_x.call_count, 2)

    def test_WhenExecuteMethodIsCalled_WhileClassWorksAsExpected_ThenMethodsAreCalledAsManyTimesAsExpected(self):
        self.nn_util.train_model = Mock(return_value=Mock())
        self.nn_util.save_keras_model = Mock(return_value=Mock())
        self.con_neural_network.execute()

        self.assertEqual(self.nn_util.train_model.call_count, 1)
        self.assertEqual(self.model.get_x.call_count, 5)
        self.assertEqual(self.nn_util.save_keras_model.call_count, 1)
