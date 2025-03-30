###############################################################################
#
# File: train.py
#
# Author: Isaac Ingram
#
# Purpose:
#
###############################################################################
import argparse
import pickle

import pandas as pd
from model import AdvancedItemWeightDetectionModel
from pathlib import Path

TIME_STR_FT = '%H:%M:%S:%f-%d-%m-%Y'

def main():
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--interaction_data', type=str, default='tmp/interaction-data.csv', help='Path to CSV with interaction data.')
    arg_parser.add_argument('--mqtt_data', type=str, default='tmp/mqtt-data.csv', help='Path to CSV with MQTT data.')
    arg_parser.add_argument('--time_delta', type=float, default=5, help='Time delta in seconds.')
    arg_parser.add_argument('--model', type=str, default='current_model.pickle', help='Path to model')
    args = arg_parser.parse_args()

    # Read interaction data and MQTT data separately
    interaction_df = pd.read_csv(args.interaction_data)
    mqtt_df = pd.read_csv(args.mqtt_data)

    # Convert timestamps from strings
    interaction_df['timestamp'] = pd.to_datetime(interaction_df.iloc[:, 0], format=TIME_STR_FT)
    mqtt_df['timestamp'] = pd.to_datetime(mqtt_df.iloc[:, 0], format=TIME_STR_FT)

    # Stored all matched data
    all_matches = []

    # Iterate through each interaction
    for index, row in interaction_df.iterrows():
        interaction_time = row['timestamp']
        target_value = row.iloc[2]  # Third column contains the target value

        # Get MQTT data within time delta after the interaction timestamp
        matching_mqtt = mqtt_df[(mqtt_df['timestamp'] >= interaction_time) &
                                (mqtt_df['timestamp'] <= interaction_time + pd.Timedelta(seconds=args.time_delta))]

        if not matching_mqtt.empty:
            # Store the matched mqtt values and the target
            match_data = {
                'mqtt_values': matching_mqtt.iloc[:, 1].tolist(),
                'target': target_value
            }
            all_matches.append(match_data)

    # Find the minimum number of mqtt values across all matches
    if all_matches:
        min_mqtt_values = min(len(match['mqtt_values']) for match in all_matches)

        # Create result dataframe with the minimum number of columns
        result_data = []
        for match in all_matches:
            data_point = {}
            # Only use up to min_mqtt_values
            for i in range(min_mqtt_values):
                if i < len(match['mqtt_values']):
                    data_point[f'w{i}'] = match['mqtt_values'][i]

            data_point['target'] = match['target']
            result_data.append(data_point)

        result_df = pd.DataFrame(result_data)

        model = AdvancedItemWeightDetectionModel(window_size=25)
        model.train(result_df)

        with open(Path.cwd() / Path(args.model), 'wb') as model_out:
            pickle.dump(model, model_out)


if __name__ == '__main__':
    main()
