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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft
from scipy.signal import find_peaks


class AdvancedItemWeightDetectionModel:
    """
    Custom model for detecting whether an item was removed or grabbed from the
    scale using just weight. Uses a random forest classifier.
    """

    _weight_history: List
    _feature_history: List
    _window_size: int

    def __init__(self, window_size: int=15):
        self._window_size = window_size
        self._weight_history = list()
        self._feature_history = list()
        self.model = RandomForestClassifier(n_estimators=200, random_state=27, max_depth=10, class_weight='balanced')
        self.scaler = StandardScaler()


    def extract_features(self, weight_series):
        """
        Extract features from a series of weight data
        :param weight_series: A series of weight data
        :return: A pandas dataframe with features for this model
        """
        features = pd.DataFrame()

        # Absolute and relative changes
        weight_change = weight_series.diff()
        features['weight_change'] = weight_change
        features['abs_weight_change'] = np.abs(weight_change)
        features['pct_weight_change'] = weight_change / weight_series.shift(1)

        # Min/max and ranges
        features['min_window_weight'] = weight_series.min()
        features['max_window_weight'] = weight_series.max()
        features['window_range'] = np.abs(features['max_window_weight'] - features['min_window_weight'])

        # Statistical features
        features['rolling_mean_5'] = weight_series.rolling(window=5).mean()
        features['rolling_mean_10'] = weight_series.rolling(window=10).mean()
        features['rolling_mean_15'] = weight_series.rolling(window=15).mean()
        features['rolling_mean_25'] = weight_series.rolling(window=25).mean()
        features['rolling_median_5'] = weight_series.rolling(window=5).median()
        features['rolling_median_10'] = weight_series.rolling(window=10).median()
        features['rolling_median_15'] = weight_series.rolling(window=15).median()
        features['rolling_median_25'] = weight_series.rolling(window=25).median()
        features['rolling_std_5'] = weight_series.rolling(window=5).std()
        features['rolling_std_10'] = weight_series.rolling(window=10).std()
        features['rolling_std_15'] = weight_series.rolling(window=15).std()
        features['rolling_std_25'] = weight_series.rolling(window=25).std()
        features['rolling_skew_10'] = weight_series.rolling(window=10).skew()
        features['rolling_kurtosis_10'] = weight_series.rolling(window=10).kurt()

        # Rate of change
        features['rate_of_change5'] = weight_change.rolling(window=5).mean()
        features['rate_of_change10'] = weight_change.rolling(window=10).mean()
        features['rate_of_change15'] = weight_change.rolling(window=15).mean()
        features['rate_of_change25'] = weight_change.rolling(window=25).mean()

        # Acceleration
        features['acceleration'] = weight_change.diff()  # Second derivative

        # Weight itself and lags
        features['weight'] = weight_series
        features['weight_lag1'] = weight_series.shift(1)
        features['weight_lag2'] = weight_series.shift(2)
        features['weight_lag5'] = weight_series.shift(5)
        features['weight_lag10'] = weight_series.shift(10)

        # Fast Fourier Transform for frequency analysis
        if len(weight_series) >= 10:
            fft_values = np.abs(fft(weight_series.values))[:len(weight_series) // 2]
            features['fft_max'] = np.max(fft_values)
            features['fft_mean'] = np.mean(fft_values)
            features['fft_std'] = np.std(fft_values)

        # Exponential moving averages (more weight to recent observations)
        features['ewm_mean_5'] = weight_series.ewm(span=5).mean()
        features['ewm_mean_10'] = weight_series.ewm(span=10).mean()

        # Crossing points between different moving averages
        features['cross_5_10'] = features['rolling_mean_5'] - features['rolling_mean_10']
        features['cross_5_15'] = features['rolling_mean_5'] - features['rolling_mean_15']
        features['cross_sign_change'] = ((features['cross_5_10'].shift(1) * features['cross_5_10']) < 0).astype(int)

        # Ratios between different windows
        features['ratio_std_5_10'] = features['rolling_std_5'] / (features['rolling_std_10'] + 1e-10)
        features['ratio_mean_5_10'] = features['rolling_mean_5'] / (features['rolling_mean_10'] + 1e-10)

        # Peaks and valleys
        weights = weight_series.to_numpy()
        peaks, _ = find_peaks(weights, height=0)
        valleys, _ = find_peaks(-weights)
        num_peaks = len(peaks)
        num_valleys = len(valleys)
        features['num_peaks'] = num_peaks
        features['num_valleys'] = num_valleys
        features['avg_peak_height'] = np.mean(weights[peaks]) if num_peaks > 0 else 0

        # Remove all non-numeric values
        features.fillna(0, inplace=True)

        return features


    def update_history(self, weight):
        """
        Add a new weight reading to the history.
        :param weight: New weight reading.
        :return:
        """
        self._weight_history.append(weight)


    def predict(self, new_weight=None):
        """
        Predict based on a new weight
        :param new_weight: The new weight value
        :return:
        """
        if new_weight is not None:
            self.update_history(new_weight)

        if len(self._weight_history) < self._window_size:
            # Not enough data to make a prediction
            return 0

        # Get the latest weight series
        weight_series = pd.Series(list(self._weight_history))

        # Extract features
        features = self.extract_features(weight_series)
        x_pred = features.iloc[-1].values.reshape(1, -1)

        # Scale features
        x_pred_scaled = self.scaler.transform(x_pred)

        # Make prediction
        return self.model.predict(x_pred_scaled)[0]


    def train(self, training_data):

        # Store x (features) and y (targets)
        x = []
        y = training_data['target'].values

        # Extract features from each row of weight data
        for index, row in training_data.iterrows():
            # Get all weight value columns
            weight_columns = [col for col in training_data.columns if col.startswith('w')]
            # Convert to weight series
            weight_series = pd.Series(row[weight_columns].values)
            # Convert to features vector
            features_df = self.extract_features(weight_series)
            feature_vector = features_df.iloc[-1].values
            x.append(feature_vector)

        x = np.array(x)
        self.scaler.fit(x)
        x_scaled = self.scaler.transform(x)
        # Split features and targets into separate train and test datasets
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=27)

        # Train the model
        print("Training model...")
        self.model.fit(x_train, y_train)
        print("Finished training.")

        # Calculate accuracy scores
        train_accuracy = self.model.score(x_train, y_train)
        test_accuracy = self.model.score(x_test, y_test)
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")

        # Get gini importance per feature
        importance_df = pd.DataFrame({
            'Feature': features_df.columns,  # Use actual feature names
            'Importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Print sorted feature importances
        print(importance_df)
