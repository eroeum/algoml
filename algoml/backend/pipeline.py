from .common import csv_to_pandas
from .common import fetch_tgz_data
from .common import split_train_test

class Pipeline():
    def __init__(self):
        self.data = None
        self.data_training = None
        self.data_testing = None

    def fetch_data(self, url, path, name):
        return fetch_tgz_data(url, path, name)

    def load_data(self, path, name):
        self.data = csv_to_pandas(path, name)
        return self.data

    def preprocess(self, data):
        self.data = data
        return data

    def split_data(self, test_ratio):
        self.data_training, self.data_testing = split_train_test(self.data, test_ratio)
        return self.data_training, self.data_testing
