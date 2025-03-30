###############################################################################
#
# File: model.py
#
# Author: Isaac Ingram
#
# Purpose:
#
###############################################################################
from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class AdvancedItemWeightDetectionModel:
    """
    Custom model for detecting whether an item was removed or grabbed from the
    scale using just weight. Uses a random forest classifier.
    """

    _weight_history: List
    _feature_history: List
    _window_size: int
    _pre_grab_window: int

    def __init__(self, window_size: int=15, pre_grab_window: int=5):
        self._window_size = window_size
        self._pre_grab_window = pre_grab_window
        self._weight_history = list()
        self._feature_history = list()


    def extract_features(self, weight_series):
        features = pd.DataFrame()

        # Absolute and relative changes
        features['weight_change'] = weight_series.diff()
        features['abs_weight_change'] = np.abs(features['weight_change'])
        features['pct_weight_change'] - features['weight_change'] / weight_series.shift(1)

        # Statistical features
        features['rolling_mean'] = weight_series.rolling(window=5).mean()
        features['rolling_std'] = weight_series.rolling(window=5).std()

        # Rate of change
        features['rate_of_change'] = features['weight_change'].rolling(window=3).mean()

        # Remove all non-numeric values
        features.fillna(0, inplace=True)


    def train(self, training_data):
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, )

