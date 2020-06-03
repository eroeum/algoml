from ... import backend as A

from joblib import dump, load
import numpy as np
import os
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class LinearRegressionPipeline(A.Pipeline):
    def __init__(self):
        self._full_data = None
        self.data = {
            'training' : {'x': None, 'y': None},
            'testing'  : {'x': None, 'y': None}
        }
        self.model = LinearRegression()
        self.imputer = SimpleImputer(strategy="median")

        self.download_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
        self.housing_path = os.path.join("datasets", "housing")
        self.housing_url = self.download_root + "datasets/housing/housing.tgz"
        self.name_tgz = "housing.tgz"
        self.name_csv = "housing.csv"

    def fetch_data(self, url=None, path=None, name=None):
        if url is None:
            url = self.housing_url
        if path is None:
            path = self.housing_path
        if name is None:
            name = self.name_tgz
        return A.fetch_tgz_data(url, path, name)

    def ingest_data(self, path=None, name=None):
        if path is None:
            path = self.housing_path
        if name is None:
            name = self.name_csv
        self._full_data = A.csv_to_pandas(path, name)
        return self._full_data

    def split_data(self, test_ratio=0.8):
        df = self._full_data.copy()
        df["income_cat"] = np.ceil(df["median_income"] / 1.5)
        df["income_cat"].where(df["income_cat"] < 5, 5.0, inplace=True)
        split = StratifiedShuffleSplit(n_splits=1, test_size=1-test_ratio, random_state=42)
        for train_index, test_index in split.split(df, df["income_cat"]):
            data_training = df.loc[train_index]
            data_testing = df.loc[test_index]
        for set_ in (data_training, data_testing):
            set_.drop("income_cat", axis=1, inplace=True)
        training_labels = data_training["median_house_value"].copy()
        data_training.drop("median_house_value", axis=1, inplace=True)
        testing_labels = data_testing["median_house_value"].copy()
        data_testing.drop("median_house_value", axis=1, inplace=True)

        self.data['training']['x'] = data_training
        self.data['training']['y'] = training_labels
        self.data['testing']['x'] = data_testing
        self.data['testing']['y'] = testing_labels

        return data_training, training_labels, data_testing, testing_labels

    def preprocess(self):
        num_attribs = list(self.data['training']['x'].drop("ocean_proximity", axis=1))
        cat_attribs = ["ocean_proximity"]

        num_pipeline = Pipeline([
            ('selector', A.DataFrameSelector(num_attribs)),
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', A.CombindedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ('selector', A.DataFrameSelector(cat_attribs)),
            ('cat_encoder', OneHotEncoder()),
        ])

        full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])

        self.data['training']['x'] = full_pipeline.fit_transform(self.data['training']['x'])
        self.data['testing']['x'] = full_pipeline.fit_transform(self.data['testing']['x'])
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
            self.save_model("linear_regression_model.pkl")

    def save_model(self, name):
        dump(self.model, name)

    def load_model(self, name):
        self.model = load(name)

    def display_score(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard Deviation", scores.std())
