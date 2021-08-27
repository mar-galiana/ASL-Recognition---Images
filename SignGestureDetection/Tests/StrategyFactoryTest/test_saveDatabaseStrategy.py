from unittest import TestCase
from unittest.mock import Mock
from Model.model import Model
from Logger.logger import Logger
from Exception.inputOutputException import InputException
from StrategyFactory.saveDatabaseStrategy import SaveDatabaseStrategy


class TestSaveDatabaseStrategy(TestCase):

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
            self.saveDatabaseStrategy = SaveDatabaseStrategy(self.logger, self.model, arguments)
            self.saveDatabaseStrategy.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 0)
        self.assertTrue(raised_exception, "Exception hasn't been raised when incorrect argument has been entered")

    def test_WhenArgumentsValueIsTrue_WhileStrategyClassWorksAsExpected_ThenWriteInfoIsCalledOnce(self):
        raised_exception = False
        arguments = ["true"]

        try:
            self.saveDatabaseStrategy = SaveDatabaseStrategy(self.logger, self.model, arguments)
            self.saveDatabaseStrategy.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 1)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")

    def test_WhenArgumentsValueIsFalse_WhileStrategyClassWorksAsExpected_ThenWriteInfoIsCalledOnce(self):
        raised_exception = False
        arguments = ["false"]

        try:
            self.saveDatabaseStrategy = SaveDatabaseStrategy(self.logger, self.model, arguments)
            self.saveDatabaseStrategy.execute()
        except InputException:
            raised_exception = True

        self.assertEqual(self.logger.write_info.call_count, 1)
        self.assertFalse(raised_exception, "Exception has been raised when correct argument has been entered")
