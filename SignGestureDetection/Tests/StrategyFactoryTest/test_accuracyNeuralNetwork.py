from unittest import TestCase
from Src.Model.model import Model
from keras.models import Sequential
from Src.Logger.logger import Logger
from unittest.mock import Mock, patch
from Src.Exception.inputOutputException import InputException
from Src.NeuralNetworks.enumerations import NeuralNetworkEnum
from Src.StrategyFactory.accuracyNeuralNetwork import AccuracyNeuralNetwork


class TestAccuracyNeuralNetwork(TestCase):

    def setUp(self):
        # Logger
        self.logger = Mock()
        self.logger.get_y = Mock(return_value=[])
        # Model
        self.model = Mock(Model)
        # Neural Network Util
        self.mock_nn_util = Mock()
        mock_sequential = Mock()
        mock_sequential.predict = Mock(return_value=[])
        self.mock_nn_util.load_keras_model = Mock(return_value=mock_sequential)
        self.mock_nn_util.get_categorical_vectors = Mock(return_value=[])

    def tearDown(self):
        self.logger = None
        self.model = None

    def test_WhenIncorrectArgumentIsEntered_WhileStrategyClassWorksAsExpected_ThenInputExceptionIsThrown(self):
        raised_exception = False
        arguments = ["nothing"]
        try:
            accuracy_neural_network = AccuracyNeuralNetwork(self.logger, self.model, self.mock_nn_util, arguments)
            accuracy_neural_network.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 0)
        self.assertTrue(raised_exception, "Exception hasn't been raised when incorrect argument has been entered")

    @patch('Src.NeuralNetworks.neuralNetwork.NeuralNetwork.resize_data', return_value=[])
    def test_WhenNNArgumentIsEntered_WhileStrategyClassWorksAsExpected_ThenNNIsCalledOnceAndWriteInfoTwice(self,
                                                                                                           mock_nn):
        raised_exception = False
        arguments = [NeuralNetworkEnum.NN.value]

        try:
            accuracy_neural_network = AccuracyNeuralNetwork(self.logger, self.model, self.mock_nn_util, arguments)
            accuracy_neural_network.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 2)
        self.assertEqual(mock_nn.call_count, 1)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")

    @patch('Src.NeuralNetworks.convolutionalNeuralNetwork.ConvolutionalNeuralNetwork.resize_data', return_value=[])
    def test_WhenCNNArgumentIsEntered_WhileStrategyClassWorksAsExpected_ThenCNNIsCalledOnceAndWriteInfoTwice(self,
                                                                                                             mock_cnn):
        raised_exception = False
        arguments = [NeuralNetworkEnum.CNN.value]

        try:
            accuracy_neural_network = AccuracyNeuralNetwork(self.logger, self.model, self.mock_nn_util, arguments)
            accuracy_neural_network.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(mock_cnn.call_count, 1)
        self.assertEqual(self.logger.write_info.call_count, 2)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")
