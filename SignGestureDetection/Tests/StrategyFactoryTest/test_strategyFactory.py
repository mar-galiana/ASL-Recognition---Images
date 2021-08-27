from unittest import TestCase
from Logger.logger import Logger
from unittest.mock import Mock
from Exception.inputOutputException import InputException
from StrategyFactory.helpStrategy import HelpStrategy
from StrategyFactory.strategyFactoy import ExecutionFactory
from StrategyFactory.trainNeuralNetworkStrategy import TrainNeuralNetworkStrategy
from StrategyFactory.saveDatabaseStrategy import SaveDatabaseStrategy
from StrategyFactory.accuracyNeuralNetworkStrategy import AccuracyNeuralNetwork


class TestExecutionFactory(TestCase):

    def setUp(self):
        self.logger = Mock(Logger)

    def tearDown(self):
        self.logger = None

    def test_WhenNoStrategyIsEntered_WhileStrategyClassWorksAsExpected_ThenInputExceptionIsThrown(self):
        raised_exception = False
        try:
            execution_factory = ExecutionFactory(self.logger, "", [])
            execution_factory.get_execution_strategy()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 0)
        self.assertTrue(raised_exception, "Exception hasn't been raised when incorrect argument has been entered")

    def test_WhenNoArgumentsAreEnteredWhenNeedIt_WhileItWorksAsExpected_ThenInputExceptionIsThrown(self):
        raised_exception = False
        try:
            execution_factory = ExecutionFactory(self.logger, "--saveDatabase", [])
            execution_factory.get_execution_strategy()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 1)
        self.assertTrue(raised_exception, "Exception hasn't been raised when incorrect argument has been entered")

    def test_WhenSaveDatabaseStrategyIsCalled_WhileItWorksAsExpected_ThenStrategyIsReturn(self):
        strategy = None
        raised_exception = False
        try:
            execution_factory = ExecutionFactory(self.logger, "--saveDatabase", ["false"])
            strategy = execution_factory.get_execution_strategy()
        except InputException:
            raised_exception = True

        self.assertTrue(isinstance(strategy, SaveDatabaseStrategy))
        self.assertEqual(self.logger.write_info.call_count, 2)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")

    def test_WhenHelpStrategyIsCalled_WhileItWorksAsExpected_ThenStrategyIsReturn(self):
        strategy = None
        raised_exception = False
        try:
            execution_factory = ExecutionFactory(self.logger, "--help", [])
            strategy = execution_factory.get_execution_strategy()
        except InputException:
            raised_exception = True

        self.assertTrue(isinstance(strategy, HelpStrategy))
        self.assertEqual(self.logger.write_info.call_count, 1)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")

    def test_WhenTrainNeuralNetworkStrategyIsCalled_WhileItWorksAsExpected_ThenStrategyIsReturn(self):
        strategy = None
        raised_exception = False
        try:
            execution_factory = ExecutionFactory(self.logger, "--trainNeuralNetwork", ["mock_arg"])
            strategy = execution_factory.get_execution_strategy()
        except InputException:
            raised_exception = True

        self.assertTrue(isinstance(strategy, TrainNeuralNetworkStrategy))
        self.assertEqual(self.logger.write_info.call_count, 2)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")

    def test_WhenAccuracyNeuralNetworkStrategyIsCalled_WhileItWorksAsExpected_ThenStrategyIsReturn(self):
        strategy = None
        raised_exception = False
        try:
            execution_factory = ExecutionFactory(self.logger, "--accuracyNeuralNetwork", ["mock_arg"])
            strategy = execution_factory.get_execution_strategy()
        except InputException:
            raised_exception = True

        self.assertTrue(isinstance(strategy, AccuracyNeuralNetwork))
        self.assertEqual(self.logger.write_info.call_count, 2)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")
