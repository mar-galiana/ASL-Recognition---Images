import numpy as np
from gesture_detection.exception.error import ParametersMissing
from gesture_detection.classification.iclassification import IClassification


class Testing(IClassification):
    NUMBER_PARAMETERS = 3

    def __init__(self):
        self.y_pred = None

    def perform(self, *information):
        if len(information) != self.NUMBER_PARAMETERS:
            raise ParametersMissing(self.NUMBER_PARAMETERS)

        data = information[0]
        sgd_clf = information[1]
        transformer = information[2]

        x_test = np.array(data.get_data()['data'])
        y_test = np.array(data.get_data()['label'])

        x_test_gray = transformer.gray.transform(x_test)
        x_test_hog = transformer.hog.transform(x_test_gray)
        x_test_prepared = transformer.scaler.transform(x_test_hog)

        self.y_pred = sgd_clf.predict(x_test_prepared)
        print(np.array(self.y_pred == y_test)[:25])
        print('')
        print('Percentage correct: ', 100*np.sum(self.y_pred == y_test)/len(y_test))

