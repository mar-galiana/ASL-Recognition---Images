from sklearn.linear_model import SGDClassifier
from gesture_detection.exception.error import ParametersMissing
from gesture_detection.classification.iclassification import IClassification


class Training(IClassification):
    NUMBER_PARAMETERS = 2

    def __init__(self):
        self.sgd_clf = None

    def perform(self, *information):
        if len(information) != self.NUMBER_PARAMETERS:
            raise ParametersMissing(NUMBER_PARAMETERS)

        x_prepared = information[0]
        y = information[1]
        self.sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        self.sgd_clf.fit(x_prepared, y)
