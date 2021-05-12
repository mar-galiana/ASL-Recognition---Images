from unittest import TestCase
from unittest.mock import Mock
from Src.Logger.Logger import Logger
from Src.StrategyFactory.helpStrategy import HelpStrategy


class TestHelpStrategy(TestCase):

    def setUp(self):
        self.logger = Mock(Logger)

    def test_WhenHelpStrategyIsCalled_WhileWorkingAsExpected_ThenWriteMessageIsCalledOnce(self):
        self.helpStrategy = HelpStrategy(self.logger)
        self.helpStrategy.execute()
        self.assertEqual(self.logger.write_message.call_count, 1)
