from ... import backend as A

from joblib import dump, load
import numpy as np
import os

from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class GradientBoostPipeline(A.Pipeline):
    def __init__(self):
        self._full_data = None
        self.data = {
            'training' : {'x': None, 'y': None},
            'testing'  : {'x': None, 'y': None}
        }
        self.model = GradientBoostingRegressor(
            max_depth=2,
            n_estimators=3,
            learning_rate=1.0
        )

        self.path = os.path.join("datasets", "iris")
        self.name_csv = 'iris.csv'

    def fetch_data(self, url=None, path=None, name=None):
        if path is None:
            path = self.path
        if name is None:
            name = self.name_csv
        df = A.fetch_pandas_dataset(lambda: datasets.load_iris(as_frame=True),
                                    path=path, name=name)
        return df

    def ingest_data(self, path=None, name=None):
        self._full_data = datasets.load_iris()
        return self._full_data

    def split_data(self, test_ratio=0.8):
        x = self._full_data['data'][:, 2:]
        y = self._full_data['target']

        split = int(len(x) * test_ratio)
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]
        shuffled_index = np.random.permutation(split)
        x_train, y_train = x_train[shuffled_index], y_train[shuffled_index]

        self.data['training']['x'] = x_train
        self.data['training']['y'] = y_train
        self.data['testing']['x'] = x_test
        self.data['testing']['y'] = y_test

        return x_train, y_train, x_test, y_test

    def preprocess(self):
        pipeline = Pipeline([
            ("scalar", StandardScaler()),
        ])

        self.data['training']['x'] = pipeline.fit_transform(self.data['training']['x'])
        self.data['testing']['x'] = pipeline.fit_transform(self.data['testing']['x'])
        return self.data

    def train(self):
        return self.model.fit(self.data['training']['x'], self.data['training']['y'])

    def evaluate(self, verbose=True):
        lin_scores = cross_val_score(self.model,
            self.data['training']['x'], self.data['training']['y'],
            scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        if verbose:
            self.display_score(lin_rmse_scores)

    def run(self, fetch_data=False, save_model=False, verbose=True):
        if fetch_data:
            self.fetch_data()
        self.ingest_data()
        self.split_data()
        self.preprocess()
        self.train()
        self.evaluate(verbose)
        if save_model:
            self.save_model("GradientBoost.pkl")

    def save_model(self, name):
        dump(self.model, name)

    def load_model(self, name):
        self.model = load(name)

    def display_score(self, scores):
        print("Mean:", scores.mean())
        print("Standard Deviation", scores.std())
