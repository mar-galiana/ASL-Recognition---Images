import numpy as np
from unittest import TestCase
from unittest.mock import Mock
from Src.Model.model import Model
from keras.models import Sequential
from Src.Logger.logger import Logger
from Src.Model.enumerations import Environment
from Src.Exception.modelException import EnvironmentException
from Src.NeuralNetworks.enumerations import NeuralNetworkEnum
from Src.NeuralNetworks.neuralNetworkUtil import NeuralNetworkUtil


class TestNeuralNetworkUtil(TestCase):

    def setUp(self):
        self.MAX_ARRAY = 10
        self.model = Mock(Model)
        self.model.get_x = Mock(return_value=np.ones((self.MAX_ARRAY, 150, 150), dtype=int))
        self.model.get_y = Mock(return_value=np.ones(self.MAX_ARRAY, dtype=int))
        self.nn_util = NeuralNetworkUtil(self.model)

    def tearDown(self):
        self.model = None

    def test_WhenGetCategoricalVectorsIsCalled_WhileClassWorksAsExpected_ThenReturnValueHasSameLengthAsTheLabels(self):
        categorical_vectors = self.nn_util.get_categorical_vectors(Environment.TRAIN, 0)
        self.assertEqual(self.model.get_y.call_count, 1)
        self.assertEqual(len(categorical_vectors), self.MAX_ARRAY)

    def test_WhenConvertToOneHotDataIsCalled_WhileClassWorksAsExpected_ThenNumberOfClassesHasToBeCorrect(self):
        n_classes = self.nn_util.convert_to_one_hot_data()
        self.assertEqual(self.model.get_y.call_count, 3)
        self.assertEqual(self.model.get_x.call_count, 2)
        self.assertEqual(n_classes, 2)

    def test_WhenTrainModelIsCalled_WhileClassWorksAsExpected_ThenMethodsHaveToBeCalledAsManyTimesAsExpected(self):
        mock_sequential = Mock(Sequential)
        return_sequential = self.nn_util.train_model(mock_sequential)

        self.assertEqual(return_sequential, mock_sequential)
        self.assertEqual(mock_sequential.fit.call_count, 1)
        self.assertEqual(mock_sequential.summary.call_count, 1)
        self.assertEqual(mock_sequential.compile.call_count, 1)

    def test_WhenSaveKerasModelIsCalled_WhileClassWorksAsExpected_ThenMethodsHaveToBeCalledAsManyTimesAsExpected(self):
        mock_sequential = Mock(Sequential)
        self.nn_util.save_keras_model(mock_sequential, NeuralNetworkEnum.NN)

        self.assertEqual(mock_sequential.save.call_count, 1)

    def test_WhenEnvironmentEnteredIsIncorrect_WhileClassWorksAsExpected_ThenEnvironmentExceptionIsRaised(self):
        exception_raised = False
        try:
            mock_sequential = Mock(Sequential)
            self.nn_util.save_keras_model(mock_sequential, "incorrect_env")
        except EnvironmentException:
            exception_raised = True

        self.assertTrue(exception_raised, "Exception hasn't been raised when incorrect argument has been entered")
