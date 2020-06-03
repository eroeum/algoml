import hashlib
import os
import numpy as np
import pandas as pd
import tarfile

from six.moves import urllib
from sklearn.datasets import fetch_openml

def csv_to_pandas(path, name):
    csv_path = os.path.join(path, name)
    return pd.read_csv(csv_path)

def fetch_open_ml_data(id, path, name):
    if not os.path.isdir(path):
        os.makedirs(path)
    csv_path = os.path.join(path, name)
    df = fetch_openml(id, as_frame=True)
    df['frame'].to_csv(csv_path, index=False, encoding='utf-8')
    return df

def fetch_pandas_dataset(bunch_func, path, name):
    if not os.path.isdir(path):
        os.makedirs(path)
    csv_path = os.path.join(path, name)
    df = bunch_func()
    df['frame'].to_csv(csv_path, index=False, encoding='utf-8')
    return df

def fetch_tgz_data(url, path, tgz_name):
    if not os.path.isdir(path):
        os.makedirs(path)
    tgz_path = os.path.join(path, tgz_name)
    urllib.request.urlretrieve(url, tgz_path)
    tgz = tarfile.open(tgz_path)
    tgz.extractall(path=path)
    tgz.close()
    return tgz_path

def pandas_to_csv(df, path, name):
    if not os.path.isdir(path):
        os.makedirs(path)
    csv_path = os.path.join(path, name)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    return df

def split_train_test(data, test_ratio=0.2, hash=hashlib.md5):
    def test_set_check(identifier, test_ratio, hash):
        return hash(np.int64(identifier)).digest()[-1] <= 256 * test_ratio
    data = data.reset_index()
    ids = data["index"]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set].drop('index', axis=1), data.loc[in_test_set].drop('index', axis=1)
