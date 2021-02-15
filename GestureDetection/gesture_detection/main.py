import numpy as np
import pandas as pd
from dataset import Dataset
from enumeration import Environment
from sklearn.metrics import confusion_matrix
from classification.transformer import Transformer
from classification.training import Training
from classification.testing import Testing


def load_dataset():
    test_data = Dataset(environment=Environment.TEST, width=80)
    train_data = Dataset(environment=Environment.TRAIN, width=80)
    return train_data, test_data


def create_confusion_matrix(y_test, y_pred):
    label_names = ['yes', 'no']
    cmx = confusion_matrix(y_test, y_pred)
    pass


def start():

    train_data, test_data = load_dataset()
    print('vamos a transformer')
    transformer = Transformer()
    transformer.perform(train_data)
    y_train = np.array(train_data.get_data()['label'])
    training = Training()
    training.perform(transformer.x_prepared, y_train)
    testing = Testing()
    testing.perform(test_data, training.sgd_clf, transformer)

    create_confusion_matrix(np.array(test_data.get_data()['label']), testing.y_pred)


if __name__ == "__main__":
    start()
