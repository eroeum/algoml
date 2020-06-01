from .. import backend as A

import copy
from joblib import dump, load
import numpy as np
import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score

class BinaryClassificationPipeline(A.Pipeline):
    def __init__(self):
        self._full_data = None
        self.data = {
            'training' : {'x': None, 'y': None},
            'testing'  : {'x': None, 'y': None}
        }
        self.model = SGDClassifier(random_state=42)

        self.id = 'mnist_784'
        self.path = os.path.join("datasets", "mnist")
        self.name_csv = 'mnist_784.csv'

    def fetch_data(self, url=None, path=None, name=None):
        if url is None:
            url = self.id
        if path is None:
            path = self.path
        if name is None:
            name = self.name_csv
        df = A.fetch_open_ml_data(url, path, name)
        return df

    def ingest_data(self):
        self._full_data =fetch_openml(self.id)
        return self._full_data

    def split_data(self, test_ratio=60000/70000):
        df = copy.copy(self._full_data)
        data, target = df['data'], df['target']
        split = int(70000 * test_ratio)
        x_train, x_test = data[:split], data[split:]
        y_train, y_test = target[:split], target[split:]
        shuffled_index = np.random.permutation(split)
        x_train, y_train = x_train[shuffled_index], y_train[shuffled_index]

        self.data['training']['x'] = x_train
        self.data['training']['y'] = (y_train == '5')
        self.data['testing']['x'] = x_test
        self.data['testing']['y'] = (y_test == '5')

        return x_train, y_train, x_test, y_test

    def preprocess(self):
        self.data = self.data
        return self.data

    def train(self):
        self.model.fit(self.data['training']['x'], self.data['training']['y'])

    def evaluate(self, verbose=True, type='confusion-matrix'):
        estimators = self.data['training']['x']
        labels = self.data['training']['y']
        if type == 'confusion-matrix':
            predictions = cross_val_predict(self.model, estimators, labels, cv=3)
            precision = precision_score(labels, predictions)
            recall = recall_score(labels, predictions)
            f1 = f1_score(labels, predictions)
            score = {'precision': precision, 'recall': recall, 'f1': f1}
        if type == 'cross-validation':
            score = {'cv': cross_val_score(self.model, estimators, labels, cv=3, scoring="accuracy")}
        if verbose:
            for key, value in score.items():
                print(key, ":", value)
        return score

    def run(self, fetch_data=False, save_model=False, verbose=True):
        if fetch_data:
            self.fetch_data()
        self.ingest_data()
        self.split_data()
        self.preprocess()
        self.train()
        self.evaluate(verbose)
        if save_model:
            self.save_model("binary_classification_model.pkl")

    def save_model(self, name):
        dump(self.model, name)

    def load_model(self, name):
        self.model = load(name)
