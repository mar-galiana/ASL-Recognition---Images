from unittest import TestCase
from Src.Model.model import Model
from Src.Logger.logger import Logger
from unittest.mock import Mock, patch
from Src.Exception.inputException import InputException
from Src.NeuralNetworks.enumerations import NeuralNetworkEnum
from Src.StrategyFactory.trainNeuralNetwork import TrainNeuralNetwork


class TestTrainNeuralNetwork(TestCase):

    def setUp(self):
        self.logger = Mock(Logger)
        self.model = Mock(Model)

    def tearDown(self):
        self.logger = None
        self.model = None

    def test_WhenIncorrectArgumentIsEntered_WhileStrategyClassWorksAsExpected_ThenInputExceptionIsThrown(self):
        raised_exception = False
        arguments = ["nothing"]
        try:
            self.trainNeuralNetwork = TrainNeuralNetwork(self.logger, self.model, arguments)
            self.trainNeuralNetwork.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 0)
        self.assertTrue(raised_exception, "Exception hasn't been raised when incorrect argument has been entered")

    @patch('Src.StrategyFactory.trainNeuralNetwork.NeuralNetwork')
    def test_WhenTrainingBasicNN_WhileStrategyClassWorksAsExpected_ThenWriteInfoIsCalledOnce(self, mock_nn):
        raised_exception = False
        arguments = [NeuralNetworkEnum.NN.value]

        try:
            self.trainNeuralNetwork = TrainNeuralNetwork(self.logger, self.model, arguments)
            self.trainNeuralNetwork.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(mock_nn.call_count, 1)
        self.assertEqual(self.logger.write_info.call_count, 1)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")

    @patch('Src.StrategyFactory.trainNeuralNetwork.ConvolutionalNeuralNetwork')
    def test_WhenTrainingCNN_WhileStrategyClassWorksAsExpected_ThenWriteInfoIsCalledOnce(self, mock_cnn):
        raised_exception = False
        arguments = [NeuralNetworkEnum.CNN.value]

        try:
            self.saveDatabaseStrategy = TrainNeuralNetwork(self.logger, self.model, arguments)
            self.saveDatabaseStrategy.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(mock_cnn.call_count, 1)
        self.assertEqual(self.logger.write_info.call_count, 1)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")
