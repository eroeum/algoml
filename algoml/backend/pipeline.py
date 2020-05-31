from .common import csv_to_pandas
from .common import fetch_tgz_data
from .common import split_train_test

from joblib import dump, load

class Pipeline():
    def __init__(self):
        self.model = None
        self._full_data = None
        self.data = {
            'training' : {'x': None, 'y': None},
            'testing'  : {'x': None, 'y': None}
        }

    def fetch_data(self, url, path, name):
        """ Fetch data locally
        @param url URL or location for data
        @param path Path to save data to
        @param name Name of resulting data
        """
        return fetch_tgz_data(url, path, name)

    def ingest_data(self, path, name):
        """ Ingest data into memory """
        self._full_data = csv_to_pandas(path, name)
        return self._full_data

    def split_data(self, test_ratio):
        """ Split data into training and testing set """
        self.data_training, self.data_testing = split_train_test(self.data, test_ratio)
        return self.data_training, self.data_testing

    def preprocess(self):
        """ Preprocess Data """
        self.data = self.data
        return self.data

    def train(self):
        """ Train model on training data """
        if not self.model:
            return None
        else:
            x = self.data['training']['x']
            y = self.data['training']['y']
            return self.model.fit(x, y)

    def evaluate(self):
        """ Evaluate model on data """
        return 0

    def run(self):
        return

    def save_model(self, name):
        """ Save model into .pkl format """
        dump(self.model, name)

    def load_model(self, name):
        """ Load .pkl model """
        self.model = load(name)
