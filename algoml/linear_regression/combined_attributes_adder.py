import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix = 3
bedrooms_ix = 4
population_ix = 5
household_ix = 6

class CombindedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_rooms = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_houshold = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_houshold, population_per_household,
                bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_houshold, population_per_household]
