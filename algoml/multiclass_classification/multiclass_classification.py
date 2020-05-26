from .. import backend as A

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score

class MulticlassClassificationPipeline(A.Pipeline):
    def __init__(self):
        super().__init__()

    def cv_score(self, estimators, labels):
        score = cross_val_score(self.model, estimators, labels, cv=3, scoring="accuracy")
        return score

    def load_data(self):
        mnist = fetch_openml('mnist_784')
        data, target = mnist["data"], mnist["target"]
        self.data = data
        self.model = SGDClassifier(random_state=42)
        self.target = target
        return self.data, self.target

    def split_data(self):
        x_train, x_test = self.data[:60000], self.data[60000:]
        y_train, y_test = self.target[:60000], self.target[60000:]
        shuffled_index = np.random.permutation(60000)
        x_train, y_train = x_train[shuffled_index], y_train[shuffled_index]
        return x_train, x_test, y_train, y_test

    def train(self, training_data, training_labels):
        self.model.fit(training_data, training_labels)
