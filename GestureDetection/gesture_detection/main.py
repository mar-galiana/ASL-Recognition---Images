import numpy as np
from dataset import Dataset
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from gesture_detection.rgb2GrayTransformer import RGB2GrayTransformer, HogTransformer


def load_dataset():
    test_data = Dataset("dataset/gesture_image_data/test", environment="test", width=80)
    train_data = Dataset("dataset/gesture_image_data/test", environment="train", width=80)
    return train_data, test_data


def processing():
    pass


def transformers(data):
    # create an instance of each transformer
    x_train = np.array(data.get_data()['data'])

    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(pixels_per_cell=(14, 14), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
    scalify = StandardScaler()

    # call fit_transform on each transform converting x_train step by step
    x_train_gray = grayify.fit_transform(x_train)
    x_train_hog = hogify.fit_transform(x_train_gray)
    x_prepared = scalify.fit_transform(x_train_hog)

    print(x_prepared.shape)
    return x_prepared, grayify, hogify, scalify


def training(x_train_prepared, y_train):
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(x_train_prepared, y_train)
    return sgd_clf


def testing(data, grayify, hogify, scalify, sgd_clf):
    x_test = np.array(data.get_data()['data'])
    y_test = np.array(data.get_data()['label'])

    x_test_gray = grayify.transform(x_test)
    x_test_hog = hogify.transform(x_test_gray)
    x_test_prepared = scalify.transform(x_test_hog)

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
    x_train_prepared, grayify, hogify, scalify = transformers(train_data)
    y_train = np.array(train_data.get_data()['label'])
    sgd_clf = training(x_train_prepared, y_train)
    testing(test_data, grayify, hogify, scalify, sgd_clf)


if __name__ == "__main__":
    start()
