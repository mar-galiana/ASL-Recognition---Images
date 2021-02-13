import numpy as np
from dataset import Dataset
from sklearn.metrics import confusion_matrix
from classification.transformer import Transformer
from classification.training import Training


def load_dataset():
    test_data = Dataset("dataset/gesture_image_data/test", environment="test", width=80)
    train_data = Dataset("dataset/gesture_image_data/test", environment="train", width=80)
    return train_data, test_data


def testing(data, transformer, sgd_clf):
    x_test = np.array(data.get_data()['data'])
    y_test = np.array(data.get_data()['label'])

    x_test_gray = transformer.gray.transform(x_test)
    x_test_hog = transformer.hog.transform(x_test_gray)
    x_test_prepared = transformer.scaler.transform(x_test_hog)

    y_pred = sgd_clf.predict(x_test_prepared)
    print(np.array(y_pred == y_test)[:25])
    print('')
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))


def create_confusion_matrix():
    label_names = ['yes', 'no']
    cmx = confusion_matrix(y_test, predictions, labels=label_names)
    df = pd.DataFrame(cmx, columns=label_names, index=label_names)
    df.columns.name = 'prediction'
    df.index.name = 'label'


def start():

    train_data, test_data = load_dataset()
    print('vamos a transformer')
    transformer = Transformer()
    transformer.perform(train_data)
    y_train = np.array(train_data.get_data()['label'])
    training = Training()
    training.perform(transformer.x_prepared, y_train)
    testing(test_data, transformer, training.sgd_clf)


if __name__ == "__main__":
    start()
