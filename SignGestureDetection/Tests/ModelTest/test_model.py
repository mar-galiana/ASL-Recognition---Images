from unittest import TestCase
from Structures.DecisionTreeLogger.logger import Logger
from unittest.mock import Mock, patch
from Model.model import Model, Image
from Model.enumerations import Environment
from ModelmodelException import EnvironmentException
from Exception.inputOutputException import PathDoesNotExistException


class TestModel(TestCase):

    def setUp(self):
        self.model = Model()

    def tearDown(self):
        self.model = None

    @patch("os.listdir", return_value=["dir1", "dir2"])
    @patch("os.path.isfile", return_value=True)
    @patch("joblib.dump", return_value=True)
    def test_WhenCreatePickle_WhileClassWorksAsExpected_ThenMethodsAreCalledAsManyTimesAsExpected(self, mock_joblib,
                                                                                                  mock_isfile,
                                                                                                  mock_listdir):
        self.model.create_pickle(True)
        self.assertEqual(mock_isfile.call_count, 4)
        self.assertEqual(mock_listdir.call_count, 2)
        self.assertEqual(mock_joblib.call_count, 2)

    def test_WhenIncorrectEnvironmentIsEntered_WhileClassWorksAsExpected_ThenEnvironmentExceptionIsRaised(self):
        exceptions_raised = False
        try:
            self.model.get_x("incorrect_environment")
        except EnvironmentException:
            exceptions_raised = True

        self.assertTrue(exceptions_raised, "Exception hasn't been raised when incorrect argument has been entered")

    @patch("os.path.exists", return_value=False)
    def test_WhenCorrectArgumentsAreEntered_WhilePicklePathDoesNotExist_ThenPathDoesNotExistExceptionIsRaised(self,
                                                                                                              mock_exists):
        exceptions_raised = False
        try:
            self.model.get_x(Environment.TRAIN)
        except PathDoesNotExistException:
            exceptions_raised = True

        self.assertEqual(mock_exists.call_count, 1)
        self.assertTrue(exceptions_raised, "Exception hasn't been raised when incorrect argument has been entered")

    @patch("os.path.exists", return_value=True)
    @patch("joblib.load", return_value={Image.DATA.value: []})
    def test_WhenCorrectArgumentsAreEntered_WhileClassWorksAsExpected_ThenMethodReturnsData(self, mock_exists,
                                                                                            mock_load):
        exceptions_raised = False
        try:
            self.model.get_x(Environment.TRAIN)
        except PathDoesNotExistException:
            exceptions_raised = True

        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(mock_exists.call_count, 1)
        self.assertFalse(exceptions_raised, "Exception hasn't been raised when incorrect argument has been entered")

    @patch("os.path.exists", return_value=True)
    @patch("joblib.load", return_value={Image.LABEL.value: []})
    def test_WhenCorrectArgumentsAreEntered_WhileClassWorksAsExpected_ThenMethodReturnsLabels(self, mock_exists,
                                                                                              mock_load):
        exceptions_raised = False
        try:
            self.model.get_y(Environment.TRAIN)
        except PathDoesNotExistException:
            exceptions_raised = True

        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(mock_exists.call_count, 1)
        self.assertFalse(exceptions_raised, "Exception hasn't been raised when incorrect argument has been entered")
