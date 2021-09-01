from unittest import TestCase
from Model.model import Model
from Logger.logger import Logger
from unittest.mock import Mock, patch
from Exception.inputOutputException import InputException
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum
from StrategyFactory.trainNeuralNetworkStrategy import TrainNeuralNetworkStrategy
from Structures.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil


class TestTrainNeuralNetwork(TestCase):

    def setUp(self):
        self.logger = Mock(Logger)
        self.model = Mock(Model)
        self.nn_util = Mock(NeuralNetworkUtil)

    def tearDown(self):
        self.logger = None
        self.model = None

    def test_WhenIncorrectArgumentIsEntered_WhileStrategyClassWorksAsExpected_ThenInputExceptionIsThrown(self):
        raised_exception = False
        arguments = ["nothing"]
        try:
            self.trainNeuralNetwork = TrainNeuralNetworkStrategy(self.logger, self.model, self.nn_util, arguments)
            self.trainNeuralNetwork.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 0)
        self.assertTrue(raised_exception, "Exception hasn't been raised when incorrect argument has been entered")

    @patch('StrategyFactory.trainNeuralNetwork.NeuralNetwork')
    def test_WhenTrainingBasicNN_WhileStrategyClassWorksAsExpected_ThenWriteInfoIsCalledOnce(self, mock_nn):
        raised_exception = False
        arguments = [NeuralNetworkTypeEnumNN.value]

        try:
            self.trainNeuralNetwork = TrainNeuralNetworkStrategy(self.logger, self.model, self.nn_util, arguments)
            self.trainNeuralNetwork.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(mock_nn.call_count, 1)
        self.assertEqual(self.logger.write_info.call_count, 1)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")

    @patch('StrategyFactory.trainNeuralNetwork.ConvolutionalNeuralNetwork')
    def test_WhenTrainingCNN_WhileStrategyClassWorksAsExpected_ThenWriteInfoIsCalledOnce(self, mock_cnn):
        raised_exception = False
        arguments = [NeuralNetworkTypeEnumCNN.value]

        try:
            self.saveDatabaseStrategy = TrainNeuralNetworkStrategy(self.logger, self.model, self.nn_util, arguments)
            self.saveDatabaseStrategy.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 1)
        self.assertEqual(mock_cnn.call_count, 1)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")
