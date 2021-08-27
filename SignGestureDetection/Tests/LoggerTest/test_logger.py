from unittest import TestCase
from from Structures.DecisionTreeLogger.logger import Logger
from unittest.mock import patch


class TestLogger(TestCase):

    def setUp(self):
        self.logger = Logger()

    def tearDown(self):
        self.logger = None

    @patch('builtins.print')
    def test_WhenWriteMessageIsCalled_WhileClassWorksAsExpected_ThenPrintFunctionIsCalledOnce(self, mock_print):
        self.logger.write_message("write message")
        self.assertEqual(mock_print.call_count, 1)

    @patch('builtins.print')
    def test_WhenWriteInfoIsCalled_WhileClassWorksAsExpected_ThenPrintFunctionIsCalledOnce(self, mock_print):
        self.logger.write_info("write info")
        self.assertEqual(mock_print.call_count, 1)

    @patch('builtins.print')
    def test_WhenWriteErrorIsCalled_WhileClassWorksAsExpected_ThenPrintFunctionIsCalledOnce(self, mock_print):
        self.logger.write_error("write error")
        self.assertEqual(mock_print.call_count, 1)
