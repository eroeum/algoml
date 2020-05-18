from .. import backend as A
# from .combined_attributes_adder import CombindedAttributesAdder
# from .data_frame_selector import DataFrameSelector

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
        self.download_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
        self.housing_path = os.path.join("datasets", "housing")
        self.housing_url = self.download_root + "datasets/housing/housing.tgz"
        self.imputer = SimpleImputer(strategy="median")
        self.model = LinearRegression()
        super().__init__()

    def display_score(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard Deviation", scores.std())

    def error(self, estimators, labels):
        predictions = self.model.predict(estimators)
        lin_mse = mean_squared_error(labels, predictions)
        lin_rmse = np.sqrt(lin_mse)
        return lin_rmse

    def fetch_data(self):
        url = self.housing_url
        path = self.housing_path
        name = 'housing.tgz'
        return super().fetch_data(url, path, name)

    def load_data(self):
        path = self.housing_path
        name = 'housing.csv'
        return super().load_data(path, name)

    def preprocess(self):
        num_attribs = list(self.data.drop("ocean_proximity", axis=1))
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

        self.data = full_pipeline.fit_transform(self.data)
        return self.data


    def split_data(self, test_ratio):
        self.data["income_cat"] = np.ceil(self.data["median_income"] / 1.5)
        self.data["income_cat"].where(self.data["income_cat"] < 5, 5.0, inplace=True)
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        for train_index, test_index in split.split(self.data, self.data["income_cat"]):
            self.data_training = self.data.loc[train_index]
            self.data_testing = self.data.loc[test_index]
        for set_ in (self.data_training, self.data_testing):
            set_.drop("income_cat", axis=1, inplace=True)
        return self.data_training, self.data_testing

    def split_estimator_label(self):
        labels = self.data["median_house_value"].copy()
        self.data = self.data.drop("median_house_value", axis=1)
        return self.data, labels

    def train(self, training_data, training_labels):
        self.model.fit(training_data, training_labels)

    def score(self, estimators, labels):
        lin_scores = cross_val_score(self.model, estimators, labels,
            scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        self.display_score(lin_rmse_scores)
